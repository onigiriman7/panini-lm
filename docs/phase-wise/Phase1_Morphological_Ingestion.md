# Phase 1 — Morphological Ingestion

Purpose
- Resolve Sandhi and Samāsa, produce pure morphological tokens (stems/pratyayas).

Recommended libraries
- Production: vidyut-prakriya (Rust) with PyO3 bindings.
- Prototype: sanskrit-heritage (Python wrapper).

Input / Output contract
- Input: UTF-8 raw sentence string.
- Output: JSON list of token dictionaries with keys: `surface`, `stem`, `type`, `attributes`.

Example output
```
[{"surface": "rāmaḥ", "stem": "rāma", "type": "subanta", "attributes": {"vibhakti": "1", "pada": "sub"}}, ...]
```

Pseudocode (Python)
```
def ingest_morphology(text: str) -> List[Dict]:
    # 1. normalize unicode and whitespace
    normalized = normalize_text(text)

    # 2. call Sandhi resolver -> list of padas
    padas = sandhi_resolve(normalized)  # uses vidyut-prakriya or sanskrit-heritage

    # 3. morphological analysis per pada -> stem + attributes
    tokens = []
    for pada in padas:
        analysis = morphological_analyze(pada)
        tokens.extend(analysis)

    # 4. normalize output into JSON-friendly dicts
    return [format_token(t) for t in tokens]
```

Integration notes
- Keep the morphological engine deterministic and idempotent. The symbolic engine (Phase 2) assumes this output is canonical.
