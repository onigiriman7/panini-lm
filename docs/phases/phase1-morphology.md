# Phase 1 — Morphological Ingestion

> Resolve Sandhi and Samāsa, produce pure morphological tokens.

---

## Overview

Phase 1 transforms raw Sanskrit text into a structured list of morphological tokens, resolving:

- **Sandhi**: Euphonic sound changes at word boundaries
- **Samāsa**: Compound word decomposition
- **Attribute extraction**: Vibhakti, vacana, puruṣa, etc.

This is the foundation for all subsequent phases.

---

## Input/Output Contract

### Input

- **Type**: `str`
- **Format**: Raw UTF-8 Sanskrit text
- **Example**: `"rāmo'pi gṛhaṃ gacchati"`

### Output

- **Type**: `Phase1Output` (see [data-contracts.md](../types/data-contracts.md))
- **Contains**: List of `MorphToken` with attributes

```python
{
    "raw_input": "rāmo'pi gṛhaṃ gacchati",
    "sandhi_splits": ["rāmaḥ", "api", "gṛham", "gacchati"],
    "tokens": [
        {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta", 
         "attributes": {"vibhakti": 1, "vacana": 1, "linga": "m"}},
        {"surface": "api", "stem": "api", "type": "avyaya", "attributes": {}},
        {"surface": "gṛham", "stem": "gṛha", "type": "subanta",
         "attributes": {"vibhakti": 2, "vacana": 1, "linga": "n"}},
        {"surface": "gacchati", "stem": "gam", "type": "tinanta",
         "attributes": {"purusa": 1, "vacana": 1, "lakara": "lat"}}
    ]
}
```

### Errors

- `SandhiResolutionError`: Ambiguous or invalid sandhi
- `UnknownTokenError`: Token not in morphological database
- `MorphologyError`: General analysis failure

---

## Dependencies

- **External**: [vidyut-prakriya](../integration/vidyut.md) (primary) or [sanskrit-heritage](../integration/sanskrit-heritage.md) (fallback)
- **Internal**: None (entry phase)

---

## Implementation Details

### Processing Pipeline

```
Raw Text
    │
    ▼
┌─────────────────┐
│ Unicode         │ Normalize to NFC, handle IAST/Devanagari
│ Normalization   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sandhi          │ Split at word boundaries
│ Resolution      │ vidyut_py.sandhi_resolve()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Samāsa          │ Decompose compound words
│ Analysis        │ (optional, configurable)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Morphological   │ Extract stem, type, attributes
│ Analysis        │ vidyut_py.morphological_analyze()
└────────┬────────┘
         │
         ▼
List[MorphToken]
```

### Pseudocode

```python
def ingest_morphology(text: str) -> Phase1Output:
    """
    Phase 1: Morphological Ingestion
    
    Deterministic transformation from raw text to structured tokens.
    """
    # 1. Normalize Unicode
    normalized = unicodedata.normalize('NFC', text.strip())
    
    # 2. Resolve sandhi (get backend with fallback)
    try:
        from vidyut_py import sandhi_resolve, morphological_analyze
        backend = "vidyut"
    except ImportError:
        from sanskrit_heritage import segment as sandhi_resolve, analyze as morphological_analyze
        backend = "heritage"
    
    padas = sandhi_resolve(normalized)
    
    # 3. Analyze each pada
    tokens = []
    for pada in padas:
        analysis = morphological_analyze(pada)
        tokens.append({
            "surface": pada,
            "stem": analysis.get("stem", pada),
            "type": analysis.get("type", "unknown"),
            "attributes": analysis.get("attributes", {})
        })
    
    return {
        "tokens": tokens,
        "raw_input": text,
        "sandhi_splits": padas
    }
```

---

## Error Handling

| Error | Cause | Recovery Strategy |
|-------|-------|-------------------|
| `SandhiResolutionError` | Ambiguous junction | Return multiple possibilities, log warning |
| `UnknownTokenError` | Out-of-vocabulary word | Mark as `<unk>`, preserve surface form |
| `ImportError` (vidyut) | Library not available | Auto-fallback to heritage |

### Graceful Degradation

```python
def safe_analyze(pada: str) -> MorphToken:
    """Analyze with graceful degradation for unknown words."""
    try:
        return morphological_analyze(pada)
    except UnknownTokenError:
        logging.warning(f"Unknown token: {pada}")
        return {
            "surface": pada,
            "stem": pada,  # Use surface as stem
            "type": "unknown",
            "attributes": {}
        }
```

---

## Test Specifications

### Unit Tests

```python
def test_sandhi_resolution():
    """Known sandhi cases should resolve correctly."""
    assert resolve_sandhi("rāmo'pi") == ["rāmaḥ", "api"]
    assert resolve_sandhi("devāśca") == ["devāḥ", "ca"]

def test_morphological_analysis():
    """Token attributes should be correct."""
    result = analyze_token("gacchati")
    assert result["stem"] == "gam"
    assert result["type"] == "tinanta"
    assert result["attributes"]["lakara"] == "lat"

def test_determinism():
    """Same input must produce identical output."""
    text = "rāmaḥ gacchati"
    assert ingest_morphology(text) == ingest_morphology(text)

def test_fallback():
    """Heritage fallback should work when vidyut unavailable."""
    with patch('vidyut_py', None):
        result = ingest_morphology("rāmaḥ")
        assert len(result["tokens"]) > 0
```

### Test Corpus

See [test-specifications.md](../testing/test-specifications.md) for the full test corpus with expected outputs.

---

## Related Documents

- [Data Contracts](../types/data-contracts.md) — `MorphToken`, `Phase1Output` definitions
- [vidyut Integration](../integration/vidyut.md) — Primary backend
- [sanskrit-heritage Integration](../integration/sanskrit-heritage.md) — Fallback backend
- [Glossary](../GLOSSARY.md) — Sandhi, Samāsa, Vibhakti definitions
- [Phase 2A](phase2a-symbolic.md) — Consumes Phase 1 output
- [Phase 2B](phase2b-neural.md) — Consumes Phase 1 output
- [Phase 5](phase5-decoding.md) — Uses Phase 1 engine for grammar constraints
