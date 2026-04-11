"""
demo.py — Terminal-runnable demonstration of the token embedding module.

Usage:
    python -m modules.token_embedding.demo

Reads the first few lines of text from data/gita.txt, analyzes each word
using vidyut, encodes the grammatical features as one-hot vectors, and
prints formatted tables showing the full pipeline.
"""

import re
import sys
import numpy as np
from pathlib import Path

from modules.token_embedding.analyzer import MorphAnalyzer, GrammaticalVector
from modules.token_embedding.embedding import (
    encode_onehot, decode_onehot, validate_onehot,
    assemble_sequence, assemble_batch,
)
from modules.token_embedding.features import (
    D_INPUT, FEATURE_ORDER, FEATURE_SIZES, FEATURE_OFFSETS, FEATURE_ENUMS,
    PrimitiveType,
)


# ===================================================================
# Text processing helpers
# ===================================================================

# Characters to strip from words (punctuation, markers)
STRIP_CHARS = set("—।,;।॥()॰॥")

def extract_words(line: str) -> list[str]:
    """Split a Devanagari line into words, removing punctuation."""
    # Split on whitespace
    tokens = line.split()
    words = []
    for token in tokens:
        # Strip leading/trailing punctuation
        cleaned = token.strip("".join(STRIP_CHARS))
        # Skip pure punctuation tokens or empty strings
        if cleaned and not all(c in STRIP_CHARS for c in cleaned):
            words.append(cleaned)
    return words


def is_chapter_header(line: str) -> bool:
    """Check if a line is a chapter header (e.g., 'अध्याय १ ...')."""
    return line.strip().startswith("अध्याय")


# ===================================================================
# Pretty-printing helpers
# ===================================================================

def print_separator(width: int = 120):
    print("─" * width)


def print_word_table(results: list[dict]):
    """Print a formatted table of analyzed words."""
    # Column widths
    col_dev = 22
    col_slp = 18
    col_type = 14
    col_features = 55

    # Header
    print(f"{'Word (देव)' :<{col_dev}} {'SLP1':<{col_slp}} {'Type':<{col_type}} {'Features':<{col_features}}")
    print_separator()

    for r in results:
        if r["vector"] is None:
            print(f"{r['devanagari']:<{col_dev}} {r['slp1']:<{col_slp}} {'??? (failed)' :<{col_type}}")
            continue

        gv = r["vector"]
        labels = gv.feature_labels()
        ptype = labels["primitive_type"]

        # Build compact feature string showing only non-NULL features
        parts = []
        for fname in FEATURE_ORDER:
            val = labels[fname]
            if val != "NULL" and fname != "primitive_type":
                parts.append(f"{fname}={val}")
        features_str = ", ".join(parts) if parts else "(no inflection)"

        print(f"{r['devanagari']:<{col_dev}} {r['slp1']:<{col_slp}} {ptype:<{col_type}} {features_str}")


def print_onehot_summary(results: list[dict]):
    """Print a compact summary of one-hot vectors."""
    print("\n╔══ One-hot encoding summary ══╗")
    print(f"  d_input = {D_INPUT}")
    print(f"  Features: {' + '.join(f'{s}' for s in FEATURE_SIZES)} = {D_INPUT}")
    print(f"  Feature names: {', '.join(FEATURE_ORDER)}")
    print()

    for r in results:
        if r["vector"] is None or r["onehot"] is None:
            continue

        vec = r["onehot"]
        ones_positions = np.where(vec == 1.0)[0].tolist()
        print(f"  {r['devanagari']:<18s} ones at positions: {ones_positions}")


def print_onehot_matrix(results: list[dict], max_words: int = 8):
    """Print the actual one-hot matrix for the first N words."""
    valid = [r for r in results if r["onehot"] is not None][:max_words]
    if not valid:
        return

    print(f"\n╔══ One-hot matrix (first {len(valid)} words, {D_INPUT} dims) ══╗")
    print(f"  Shape: ({len(valid)}, {D_INPUT})")
    print()

    # Print feature block headers
    header_parts = []
    for name, size in zip(FEATURE_ORDER, FEATURE_SIZES):
        abbrev = name[:3].upper()
        header_parts.append(f"{abbrev:^{size}s}")
    print(f"  {'':18s} {'|'.join(header_parts)}")
    print_separator(18 + D_INPUT + len(FEATURE_SIZES))

    for r in valid:
        vec = r["onehot"]
        # Format: one character per dimension, split by feature
        blocks = []
        for offset, size in zip(FEATURE_OFFSETS, FEATURE_SIZES):
            slot = vec[offset : offset + size]
            block = "".join("█" if v == 1.0 else "·" for v in slot)
            blocks.append(block)
        row_str = "|".join(blocks)
        print(f"  {r['devanagari']:<18s} {row_str}")


def print_roundtrip_test(results: list[dict]):
    """Demonstrate encode→decode round-trip on a few words."""
    print("\n╔══ Round-trip validation (encode → decode) ══╗")
    valid = [r for r in results if r["vector"] is not None][:5]

    all_pass = True
    for r in valid:
        original = r["vector"]
        encoded = encode_onehot(original)
        decoded = decode_onehot(encoded)

        match = original.as_tuple() == decoded.as_tuple()
        status = "✓ PASS" if match else "✗ FAIL"
        if not match:
            all_pass = False
        print(f"  {r['devanagari']:<18s} {status}")

    if all_pass:
        print(f"\n  All {len(valid)} round-trip tests passed.")


def print_sequence_demo(results: list[dict], window_size: int = 8):
    """Demonstrate context window assembly."""
    vectors = [r["vector"] for r in results if r["vector"] is not None]
    if not vectors:
        return

    print(f"\n╔══ Context window assembly (W={window_size}) ══╗")

    seq_matrix = assemble_sequence(vectors, window_size)
    print(f"  assemble_sequence({len(vectors)} vectors, W={window_size})")
    print(f"  → shape: {seq_matrix.shape}")
    print(f"  → dtype: {seq_matrix.dtype}")
    print(f"  → total ones: {int(seq_matrix.sum())} (expected: {window_size} × 9 = {window_size * 9})")

    # Batch demo
    if len(vectors) >= 4:
        seq1 = vectors[:4]
        seq2 = vectors[2:6] if len(vectors) >= 6 else vectors[:4]
        batch_matrix = assemble_batch([seq1, seq2], window_size=4)
        print(f"\n  assemble_batch(B=2, W=4)")
        print(f"  → shape: {batch_matrix.shape}  (B=2, W=4, d_input={D_INPUT})")


# ===================================================================
# Main demo
# ===================================================================

def main():
    print("=" * 80)
    print("  Pāṇinian Token Embedding — Demo")
    print("  Converting Sanskrit Gītā text → 62-dim grammatical feature vectors")
    print("=" * 80)

    # Initialize analyzer
    print("\nInitializing vidyut morphological analyzer...")
    analyzer = MorphAnalyzer()
    print("  ✓ Kosha (word lookup) loaded")
    print("  ✓ Chedaka (segmenter fallback) loaded")

    # Read gita.txt
    gita_path = Path("data/gita.txt")
    if not gita_path.exists():
        print(f"ERROR: {gita_path} not found. Run from the project root.", file=sys.stderr)
        sys.exit(1)

    lines = gita_path.read_text(encoding="utf-8").splitlines()

    # Select first 3 text lines (skip chapter headers)
    text_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if is_chapter_header(line):
            continue
        text_lines.append(line)
        if len(text_lines) >= 3:
            break

    # Process each line
    all_results = []
    for line_num, line in enumerate(text_lines, 1):
        print(f"\n{'━' * 80}")
        print(f"  Line {line_num}: {line[:80]}{'...' if len(line) > 80 else ''}")
        print(f"{'━' * 80}")

        words = extract_words(line)
        results = []

        for word_dev in words:
            word_slp1 = analyzer.transliterate(word_dev)
            gv = analyzer.analyze(word_dev)

            onehot = None
            if gv is not None:
                onehot = encode_onehot(gv)
                validate_onehot(onehot)

            results.append({
                "devanagari": word_dev,
                "slp1": word_slp1,
                "vector": gv,
                "onehot": onehot,
            })

        print_word_table(results)
        print_onehot_summary(results)
        print_onehot_matrix(results)
        print_roundtrip_test(results)
        print_sequence_demo(results)

        all_results.extend(results)

    # Final summary
    total = len(all_results)
    analyzed = sum(1 for r in all_results if r["vector"] is not None)
    failed = total - analyzed

    type_counts = {}
    for r in all_results:
        if r["vector"] is not None:
            ptype = PrimitiveType(r["vector"].primitive_type).name
            type_counts[ptype] = type_counts.get(ptype, 0) + 1

    print(f"\n{'=' * 80}")
    print(f"  Summary")
    print(f"{'=' * 80}")
    print(f"  Total words processed : {total}")
    print(f"  Successfully analyzed : {analyzed} ({100*analyzed/total:.1f}%)")
    print(f"  Failed to analyze     : {failed} ({100*failed/total:.1f}%)")
    print(f"  Primitive type counts : {type_counts}")
    print(f"  d_input               : {D_INPUT}")
    print(f"  Feature sizes         : {FEATURE_SIZES}")
    print()


if __name__ == "__main__":
    main()
