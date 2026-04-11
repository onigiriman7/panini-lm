"""
generate_report.py — Detailed report of vidyut raw output vs. token embedding module output.

Samples 10 lines from data/gita.txt, runs them through the pipeline, and
generates an HTML report for easy visualization.
"""

import json
import html as html_lib
import re
import numpy as np
from pathlib import Path

from vidyut.kosha import Kosha
from vidyut.lipi import transliterate, Scheme

from modules.token_embedding.analyzer import MorphAnalyzer, GrammaticalVector
from modules.token_embedding.embedding import encode_onehot, validate_onehot
from modules.token_embedding.features import (
    D_INPUT, FEATURE_ORDER, FEATURE_SIZES, FEATURE_OFFSETS, FEATURE_ENUMS,
)


# ── Text helpers ──────────────────────────────────────────────────
STRIP_CHARS = set("—।,;।॥()॰॥")

def extract_words(line: str) -> list[str]:
    tokens = line.split()
    words = []
    for token in tokens:
        cleaned = token.strip("".join(STRIP_CHARS))
        if cleaned and not all(c in STRIP_CHARS for c in cleaned):
            words.append(cleaned)
    return words

def is_chapter_header(line: str) -> bool:
    return line.strip().startswith("अध्याय")


# ── Collect raw vidyut data for a word ────────────────────────────
def collect_vidyut_raw(kosha: Kosha, word_slp1: str) -> list[dict]:
    """Return a list of dicts, one per kosha entry, with raw vidyut fields."""
    try:
        entries = kosha[word_slp1]
    except KeyError:
        return []

    results = []
    for e in entries:
        d = {"type": type(e).__name__, "is_avyaya": e.is_avyaya, "lemma": str(e.lemma)}
        if type(e).__name__ == "PyPadaEntry_Tinanta":
            d["lakara"] = str(e.lakara)
            d["purusha"] = str(e.purusha)
            d["vacana"] = str(e.vacana)
            d["prayoga"] = str(e.prayoga)
            if e.dhatu_entry:
                de = e.dhatu_entry
                d["dhatu_clean_text"] = str(de.clean_text)
                d["dhatu_artha_en"] = str(de.artha_en) if de.artha_en else ""
                d["dhatu_aupadeshika"] = str(de.dhatu.aupadeshika)
                d["dhatu_gana"] = str(de.dhatu.gana)
                d["dhatu_pada"] = str(de.pada) if de.pada else "None"
                d["dhatu_prefixes"] = [str(p) for p in de.dhatu.prefixes] if de.dhatu.prefixes else []
        elif type(e).__name__ == "PyPadaEntry_Subanta":
            d["vibhakti"] = str(e.vibhakti)
            d["vacana"] = str(e.vacana)
            d["linga"] = str(e.linga)
        results.append(d)
    return results


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("Loading vidyut and token embedding module...")
    analyzer = MorphAnalyzer()
    kosha = Kosha("vidyut-data/kosha")

    # Read gita.txt, sample 10 non-header, non-empty lines
    lines = Path("data/gita.txt").read_text(encoding="utf-8").splitlines()
    text_lines = []
    for line in lines:
        line = line.strip()
        if not line or is_chapter_header(line):
            continue
        text_lines.append(line)
        if len(text_lines) >= 10:
            break

    print(f"Selected {len(text_lines)} lines from Gītā.")

    # ── Process each line ─────────────────────────────────────────
    all_line_data = []  # list of { line_text, line_num, words: [...] }

    for line_num, line in enumerate(text_lines, 1):
        words = extract_words(line)
        word_data_list = []

        for word_dev in words:
            word_slp1 = transliterate(word_dev, Scheme.Devanagari, Scheme.Slp1)

            # 1) Raw vidyut output
            raw_entries = collect_vidyut_raw(kosha, word_slp1)

            # 2) Token embedding module output
            gv = analyzer.analyze(word_dev)
            onehot = None
            feature_labels = None
            ones_positions = None
            if gv is not None:
                onehot = encode_onehot(gv)
                validate_onehot(onehot)
                feature_labels = gv.feature_labels()
                ones_positions = np.where(onehot == 1.0)[0].tolist()

            word_data_list.append({
                "devanagari": word_dev,
                "slp1": word_slp1,
                "vidyut_raw_count": len(raw_entries),
                "vidyut_raw": raw_entries,
                "gv": gv,
                "feature_labels": feature_labels,
                "ones_positions": ones_positions,
                "onehot": onehot,
            })

        all_line_data.append({
            "line_text": line,
            "line_num": line_num,
            "words": word_data_list,
        })

    # ── Generate HTML report ──────────────────────────────────────
    html = build_html_report(all_line_data)
    out_path = Path("token_embedding_report.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"\nReport written to {out_path}")
    print(f"  Total lines: {len(all_line_data)}")
    total_words = sum(len(ld['words']) for ld in all_line_data)
    analyzed = sum(1 for ld in all_line_data for w in ld['words'] if w['gv'] is not None)
    print(f"  Total words: {total_words}")
    print(f"  Successfully analyzed: {analyzed} ({100*analyzed/total_words:.1f}%)")
    print(f"  Failed: {total_words - analyzed}")


# ── HTML Report Builder ───────────────────────────────────────────
def build_html_report(all_line_data: list[dict]) -> str:
    """Build a self-contained HTML report."""

    # Statistics
    total_words = sum(len(ld['words']) for ld in all_line_data)
    analyzed = sum(1 for ld in all_line_data for w in ld['words'] if w['gv'] is not None)
    failed = total_words - analyzed

    # Count by type
    type_counts = {"DHAATU": 0, "PRAATIPADIKA": 0, "AVYAYA": 0, "FAILED": 0}
    for ld in all_line_data:
        for w in ld['words']:
            if w['feature_labels']:
                pt = w['feature_labels']['primitive_type']
                type_counts[pt] = type_counts.get(pt, 0) + 1
            else:
                type_counts["FAILED"] += 1

    # Feature distribution
    feature_dist = {fname: {} for fname in FEATURE_ORDER}
    for ld in all_line_data:
        for w in ld['words']:
            if w['feature_labels']:
                for fname in FEATURE_ORDER:
                    val = w['feature_labels'][fname]
                    feature_dist[fname][val] = feature_dist[fname].get(val, 0) + 1

    # Build word detail rows
    word_rows_html = ""
    for ld in all_line_data:
        # Line header
        line_text_esc = html_lib.escape(ld['line_text'][:120])
        word_rows_html += f"""
        <tr class="line-header">
            <td colspan="6">
                <strong>Line {ld['line_num']}:</strong> {line_text_esc}{'…' if len(ld['line_text']) > 120 else ''}
            </td>
        </tr>"""

        for w in ld['words']:
            dev = html_lib.escape(w['devanagari'])
            slp = html_lib.escape(w['slp1'])

            # Vidyut raw column
            if w['vidyut_raw']:
                raw_parts = []
                for i, entry in enumerate(w['vidyut_raw'][:4]):  # show max 4
                    etype = entry['type'].replace('PyPadaEntry_', '')
                    parts_inner = [f"<b>{etype}</b> (lemma: {html_lib.escape(entry['lemma'])})"]
                    if etype == "Tinanta":
                        parts_inner.append(f"lakāra={entry.get('lakara','')}, puruṣa={entry.get('purusha','')}")
                        parts_inner.append(f"vacana={entry.get('vacana','')}, prayoga={entry.get('prayoga','')}")
                        if entry.get('dhatu_clean_text'):
                            parts_inner.append(f"dhātu={entry['dhatu_clean_text']} ({entry.get('dhatu_artha_en','')})")
                            if entry.get('dhatu_prefixes'):
                                parts_inner.append(f"prefixes={entry['dhatu_prefixes']}")
                    elif etype == "Subanta":
                        parts_inner.append(f"vibhakti={entry.get('vibhakti','')}, vacana={entry.get('vacana','')}, liṅga={entry.get('linga','')}")
                    raw_parts.append("<br>".join(parts_inner))
                raw_html = "<hr class='entry-sep'>".join(raw_parts)
                if w['vidyut_raw_count'] > 4:
                    raw_html += f"<br><i>... +{w['vidyut_raw_count'] - 4} more entries</i>"
                raw_count_badge = f'<span class="badge">{w["vidyut_raw_count"]} entries</span>'
            else:
                raw_html = '<span class="no-data">No kosha entry</span>'
                raw_count_badge = '<span class="badge badge-fail">0</span>'

            # Module output column
            if w['feature_labels']:
                fl = w['feature_labels']
                pt = fl['primitive_type']
                pt_class = pt.lower()
                module_parts = [f'<span class="ptype ptype-{pt_class}">{pt}</span>']
                feat_items = []
                for fname in FEATURE_ORDER:
                    val = fl[fname]
                    if fname == 'primitive_type':
                        continue
                    css = "feat-null" if val == "NULL" else "feat-active"
                    feat_items.append(f'<span class="{css}">{fname}=<b>{val}</b></span>')
                module_parts.append(", ".join(feat_items))
                module_html = "<br>".join(module_parts)
            else:
                module_html = '<span class="no-data">Analysis failed</span>'

            # One-hot visualization
            if w['onehot'] is not None:
                onehot_html = build_onehot_viz(w['onehot'])
            else:
                onehot_html = '<span class="no-data">—</span>'

            row_class = "row-ok" if w['gv'] else "row-fail"
            word_rows_html += f"""
        <tr class="{row_class}">
            <td class="dev-word">{dev}</td>
            <td class="slp-word">{slp}</td>
            <td class="raw-col">{raw_count_badge}{raw_html}</td>
            <td class="module-col">{module_html}</td>
            <td class="onehot-col">{onehot_html}</td>
        </tr>"""

    # Feature distribution HTML
    feat_dist_html = ""
    for fname in FEATURE_ORDER:
        dist = feature_dist[fname]
        if not dist:
            continue
        items = sorted(dist.items(), key=lambda x: -x[1])
        bars = ""
        max_count = max(dist.values()) if dist else 1
        for val, count in items:
            pct = 100 * count / total_words
            bar_w = max(4, int(200 * count / max_count))
            color = "#94a3b8" if val == "NULL" else "#3b82f6"
            bars += f'<div class="dist-row"><span class="dist-label">{val}</span><div class="dist-bar" style="width:{bar_w}px;background:{color}"></div><span class="dist-count">{count} ({pct:.0f}%)</span></div>'
        feat_dist_html += f'<div class="feat-block"><h4>{fname}</h4>{bars}</div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pāṇinian Token Embedding — Detailed Report</title>
<style>
  :root {{ --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff; --border: #30363d;
           --card-bg: #161b22; --success: #3fb950; --fail: #f85149; --warn: #d29922; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
          background: var(--bg); color: var(--fg); line-height: 1.6; padding: 2rem; }}
  h1 {{ color: var(--accent); margin-bottom: 0.5rem; font-size: 1.8rem; }}
  h2 {{ color: var(--accent); margin: 2rem 0 1rem; font-size: 1.3rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }}
  h3 {{ color: #8b949e; margin: 1.5rem 0 0.5rem; font-size: 1.1rem; }}
  h4 {{ color: #8b949e; margin: 0.8rem 0 0.3rem; font-size: 0.95rem; }}
  .subtitle {{ color: #8b949e; margin-bottom: 2rem; }}

  /* Summary cards */
  .summary {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }}
  .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
           padding: 1rem 1.5rem; min-width: 160px; }}
  .card .num {{ font-size: 2rem; font-weight: bold; color: var(--accent); }}
  .card .label {{ color: #8b949e; font-size: 0.85rem; }}
  .card.success .num {{ color: var(--success); }}
  .card.fail .num {{ color: var(--fail); }}

  /* Type distribution */
  .type-bar {{ display: flex; height: 32px; border-radius: 6px; overflow: hidden; margin: 0.5rem 0; }}
  .type-bar div {{ display: flex; align-items: center; justify-content: center;
                   font-size: 0.75rem; font-weight: bold; color: #fff; }}
  .type-dhaatu {{ background: #2563eb; }}
  .type-praatipadika {{ background: #16a34a; }}
  .type-avyaya {{ background: #d97706; }}
  .type-failed {{ background: #dc2626; }}

  /* Main table */
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.85rem; }}
  th {{ background: var(--card-bg); color: var(--accent); padding: 0.7rem; text-align: left;
        border-bottom: 2px solid var(--border); position: sticky; top: 0; z-index: 10; }}
  td {{ padding: 0.5rem 0.7rem; border-bottom: 1px solid var(--border); vertical-align: top; }}
  .line-header td {{ background: #1c2333; color: #8b949e; font-size: 0.9rem; padding: 0.6rem; }}
  .row-fail {{ background: rgba(248, 81, 73, 0.05); }}
  .dev-word {{ font-size: 1.1rem; font-weight: bold; white-space: nowrap; }}
  .slp-word {{ font-family: monospace; color: #8b949e; white-space: nowrap; }}
  .raw-col {{ max-width: 350px; font-size: 0.8rem; }}
  .module-col {{ max-width: 350px; }}
  .onehot-col {{ min-width: 250px; }}

  .badge {{ display: inline-block; background: var(--accent); color: #000; font-size: 0.7rem;
            padding: 1px 6px; border-radius: 10px; margin-bottom: 4px; font-weight: bold; }}
  .badge-fail {{ background: var(--fail); color: #fff; }}
  .entry-sep {{ border: none; border-top: 1px dashed var(--border); margin: 4px 0; }}
  .no-data {{ color: #6e7681; font-style: italic; }}

  .ptype {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: bold;
            font-size: 0.75rem; margin-bottom: 4px; }}
  .ptype-dhaatu {{ background: #1d4ed8; color: #fff; }}
  .ptype-praatipadika {{ background: #15803d; color: #fff; }}
  .ptype-avyaya {{ background: #b45309; color: #fff; }}
  .feat-null {{ color: #6e7681; }}
  .feat-active {{ color: #e2e8f0; }}

  /* One-hot mini viz */
  .onehot-grid {{ display: flex; gap: 1px; flex-wrap: nowrap; }}
  .oh-block {{ display: flex; gap: 0; }}
  .oh-cell {{ width: 4px; height: 16px; }}
  .oh-on {{ background: #3b82f6; }}
  .oh-off {{ background: #1e293b; }}
  .oh-sep {{ width: 2px; height: 16px; background: var(--border); }}
  .oh-labels {{ display: flex; gap: 1px; font-size: 0.55rem; color: #6e7681; margin-top: 2px; }}
  .oh-label {{ text-align: center; }}

  /* Feature distribution */
  .feat-dist-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }}
  .feat-block {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem; }}
  .dist-row {{ display: flex; align-items: center; gap: 0.5rem; margin: 2px 0; }}
  .dist-label {{ width: 100px; text-align: right; font-size: 0.8rem; color: #8b949e; font-family: monospace; }}
  .dist-bar {{ height: 14px; border-radius: 3px; }}
  .dist-count {{ font-size: 0.75rem; color: #6e7681; }}

  /* Pipeline diagram */
  .pipeline {{ display: flex; align-items: center; gap: 0; margin: 1.5rem 0; flex-wrap: wrap; }}
  .pipe-step {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
                padding: 0.8rem 1.2rem; text-align: center; min-width: 140px; }}
  .pipe-step .step-title {{ font-weight: bold; color: var(--accent); font-size: 0.9rem; }}
  .pipe-step .step-desc {{ font-size: 0.75rem; color: #8b949e; margin-top: 0.3rem; }}
  .pipe-arrow {{ font-size: 1.5rem; color: var(--border); padding: 0 0.3rem; }}
</style>
</head>
<body>

<h1>Pāṇinian Token Embedding — Detailed Pipeline Report</h1>
<p class="subtitle">10 sample lines from Bhagavad Gītā · vidyut raw output vs. token embedding module output</p>

<!-- Pipeline diagram -->
<h2>Pipeline Overview</h2>
<div class="pipeline">
  <div class="pipe-step">
    <div class="step-title">Input</div>
    <div class="step-desc">Devanagari word<br><code>धर्मक्षेत्रे</code></div>
  </div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step">
    <div class="step-title">Transliterate</div>
    <div class="step-desc">vidyut.lipi<br>Devanagari → SLP1</div>
  </div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step">
    <div class="step-title">Vidyut Kosha</div>
    <div class="step-desc">Raw morphological<br>lookup (N entries)</div>
  </div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step">
    <div class="step-title">Disambiguation</div>
    <div class="step-desc">Pick best entry<br>(avyaya &gt; tiṅanta &gt; subanta)</div>
  </div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step">
    <div class="step-title">GrammaticalVector</div>
    <div class="step-desc">9 features<br>(int indices)</div>
  </div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step">
    <div class="step-title">One-Hot Encoding</div>
    <div class="step-desc">62-dim float32<br>vector</div>
  </div>
</div>

<!-- Summary stats -->
<h2>Summary Statistics</h2>
<div class="summary">
  <div class="card"><div class="num">{len(all_line_data)}</div><div class="label">Lines processed</div></div>
  <div class="card"><div class="num">{total_words}</div><div class="label">Total words</div></div>
  <div class="card success"><div class="num">{analyzed}</div><div class="label">Successfully analyzed</div></div>
  <div class="card fail"><div class="num">{failed}</div><div class="label">Failed to analyze</div></div>
  <div class="card"><div class="num">{D_INPUT}</div><div class="label">d_input (one-hot dims)</div></div>
  <div class="card"><div class="num">{100*analyzed/total_words:.1f}%</div><div class="label">Coverage</div></div>
</div>

<!-- Type distribution bar -->
<h3>Primitive Type Distribution</h3>
<div class="type-bar" style="width:100%;max-width:600px">
  {"".join(f'<div class="type-{k.lower()}" style="flex:{v}">{k} ({v})</div>' for k, v in type_counts.items() if v > 0)}
</div>

<!-- Feature distributions -->
<h2>Feature Value Distributions</h2>
<div class="feat-dist-grid">
{feat_dist_html}
</div>

<!-- Detailed word-by-word table -->
<h2>Word-by-Word Detail: Vidyut Raw vs. Module Output</h2>
<p style="color:#8b949e;font-size:0.85rem;margin-bottom:0.5rem;">
  Each row shows: the raw vidyut kosha entries (all ambiguities) → the module's disambiguated GrammaticalVector → the 62-dim one-hot encoding.
</p>
<table>
<thead>
<tr>
  <th>Word (देव)</th>
  <th>SLP1</th>
  <th>Vidyut Raw (Kosha Entries)</th>
  <th>Module Output (GrammaticalVector)</th>
  <th>One-Hot (62 dims)</th>
</tr>
</thead>
<tbody>
{word_rows_html}
</tbody>
</table>

<h2>Encoding Reference</h2>
<table style="max-width:700px">
<thead><tr><th>Feature</th><th>Offset</th><th>Size</th><th>Values</th></tr></thead>
<tbody>
{"".join(f'<tr><td><b>{fname}</b></td><td>{off}</td><td>{sz}</td><td style="font-size:0.75rem">{", ".join(e.name for e in enum)}</td></tr>' for fname, off, sz, enum in zip(FEATURE_ORDER, FEATURE_OFFSETS, FEATURE_SIZES, FEATURE_ENUMS))}
</tbody>
</table>

<p style="color:#6e7681;margin-top:2rem;font-size:0.8rem;">Generated by generate_report.py · panini-lm token embedding module</p>
</body>
</html>"""

    return html


def build_onehot_viz(vec: np.ndarray) -> str:
    """Build a tiny inline one-hot visualization as HTML spans."""
    parts = []
    for fname, offset, size in zip(FEATURE_ORDER, FEATURE_OFFSETS, FEATURE_SIZES):
        cells = ""
        for i in range(size):
            cls = "oh-on" if vec[offset + i] == 1.0 else "oh-off"
            cells += f'<div class="oh-cell {cls}"></div>'
        parts.append(f'<div class="oh-block">{cells}</div><div class="oh-sep"></div>')
    grid = f'<div class="onehot-grid">{"".join(parts)}</div>'
    # labels
    labels = ""
    for fname, size in zip(FEATURE_ORDER, FEATURE_SIZES):
        abbr = fname[:3]
        w = size * 4 + (size - 1) + 2  # approx width
        labels += f'<span class="oh-label" style="width:{w}px">{abbr}</span>'
    return grid + f'<div class="oh-labels">{labels}</div>'


if __name__ == "__main__":
    main()
