# sanskrit-heritage Integration

> Pure Python Sanskrit analysis toolkit (fallback for vidyut-prakriya).

---

## Overview

**sanskrit-heritage** is a Python library for Sanskrit text processing, providing sandhi resolution and morphological analysis.

- **Use case**: Prototyping, fallback when vidyut unavailable
- **Language**: Pure Python
- **Performance**: Slower than vidyut, but easier to debug

---

## Installation

```bash
pip install sanskrit-heritage

# Or from source
git clone https://github.com/sanskrit-heritage/sanskrit-heritage.git
cd sanskrit-heritage
pip install -e .
```

---

## API Reference

### `segment(text: str) -> List[str]`

Split text by resolving sandhi.

```python
from sanskrit_heritage import segment

result = segment("rāmo'pi gṛhaṃ gacchati")
# ['rāmaḥ', 'api', 'gṛham', 'gacchati']
```

### `analyze(pada: str) -> Dict`

Morphological analysis of a single word.

```python
from sanskrit_heritage import analyze

result = analyze("gacchati")
# {
#     "stem": "gam",
#     "type": "tinanta",
#     "attributes": {"lakara": "lat", "purusa": 1, "vacana": 1}
# }
```

---

## Panini-LM Integration

### Fallback Implementation

```python
def heritage_ingest(text: str) -> Phase1Output:
    """
    Phase 1 using sanskrit-heritage (fallback).
    """
    from sanskrit_heritage import segment, analyze
    
    padas = segment(text)
    tokens = [
        {
            "surface": pada,
            "stem": analyze(pada).get("stem", pada),
            "type": analyze(pada).get("type", "unknown"),
            "attributes": analyze(pada).get("attributes", {})
        }
        for pada in padas
    ]
    
    return {
        "tokens": tokens,
        "raw_input": text,
        "sandhi_splits": padas
    }
```

---

## Comparison with vidyut

| Feature | vidyut-prakriya | sanskrit-heritage |
|---------|-----------------|-------------------|
| Language | Rust + PyO3 | Pure Python |
| Speed | Fast (~10x) | Slower |
| Installation | Requires Rust | pip install |
| Debugging | Harder | Easier |
| Best for | Production | Prototyping |

---

## Related Documentation

- [vidyut](vidyut.md) — Primary implementation
- [Phase 1 — Morphology](../phases/phase1-morphology.md)
