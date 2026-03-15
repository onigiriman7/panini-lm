# Phase 1 — Morphological Ingestion

> Resolve Sandhi and Samāsa, produce pure morphological tokens, and factorize for Phase 2B.

---

## Overview

Phase 1 transforms raw Sanskrit text into a structured list of morphological tokens, resolving:

- **Sandhi**: Euphonic sound changes at word boundaries
- **Samāsa**: Compound word decomposition
- **Attribute extraction**: Vibhakti, vacana, puruṣa, etc.
- **Factorization**: Convert tokens to parallel ID tensors for Phase 2B

This is the foundation for all subsequent phases.

### Key Innovation: Factorized Output

Phase 1 not only produces `MorphToken` objects, but also generates `FactorizedTokenBatch` — the 5 parallel ID tensors required by Phase 2B's factorized embedding architecture:

```
MorphToken → root_ids, type_ids, vibhakti_ids, vacana_ids, purusa_ids
```

This enables:
- **Zero OOV**: Any valid inflection maps to known IDs
- **12× embedding reduction**: ~4,000 vocabulary vs 50,000+

---

## Input/Output Contract

### Input

- **Type**: `str`
- **Format**: Raw UTF-8 Sanskrit text
- **Example**: `"rāmo'pi gṛhaṃ gacchati"`

### Output

- **Type**: `Phase1Output` (see [data-contracts.md](../types/data-contracts.md))
- **Contains**: List of `MorphToken` with attributes AND `FactorizedTokenBatch`

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
    ],
    # NEW: Factorized tensors for Phase 2B (Zero OOV architecture)
    "factorized": {
        "root_ids": [100, 5, 101, 6],      # rāma, api, gṛha, gam
        "type_ids": [0, 2, 0, 1],          # subanta, avyaya, subanta, tiṅanta
        "vibhakti_ids": [1, 0, 2, 0],      # nom, none, acc, none
        "vacana_ids": [1, 0, 1, 1],        # sing, none, sing, sing
        "purusa_ids": [0, 0, 0, 1]         # none, none, none, 3rd
    }
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
┌─────────────────┐
│ FACTORIZATION   │ Convert to parallel ID tensors
│ (NEW)           │ root_ids, type_ids, vibhakti_ids, etc.
└────────┬────────┘
         │
         ▼
Phase1Output (tokens + factorized)
    │
    ├───► Phase 2A (tokens → Matrix M)
    │
    └───► Phase 2B (factorized → Embeddings)
```

### Pseudocode

```python
# ID mappings for factorization (see data-contracts.md)
TYPE_TO_ID = {"subanta": 0, "tinanta": 1, "avyaya": 2, "krdanta": 3, "taddhita": 4, "samasa": 5, "none": 6}
VIBHAKTI_TO_ID = {"none": 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, "vocative": 8}
VACANA_TO_ID = {"none": 0, 1: 1, 2: 2, 3: 3}
PURUSA_TO_ID = {"none": 0, 1: 1, 2: 2, 3: 3}


def factorize_tokens(tokens: List[MorphToken], root_vocab: Dict) -> FactorizedTokenBatch:
    """
    Convert MorphTokens to parallel ID tensors for Phase 2B factorized embeddings.
    
    This is THE KEY STEP that enables:
    - Zero OOV: Any valid inflection maps to known component IDs
    - 12× embedding parameter reduction
    """
    root_ids, type_ids, vibhakti_ids, vacana_ids, purusa_ids = [], [], [], [], []
    
    for token in tokens:
        # Root/stem is the semantic core (~4000 vocabulary)
        root_ids.append(root_vocab.get(token["stem"], 1))  # 1 = UNK
        
        # Grammatical dimensions (tiny vocabularies)
        type_ids.append(TYPE_TO_ID.get(token["type"], 6))
        attrs = token.get("attributes", {})
        vibhakti_ids.append(VIBHAKTI_TO_ID.get(attrs.get("vibhakti"), 0))
        vacana_ids.append(VACANA_TO_ID.get(attrs.get("vacana"), 0))
        purusa_ids.append(PURUSA_TO_ID.get(attrs.get("purusa"), 0))
    
    return {
        "root_ids": root_ids,
        "type_ids": type_ids,
        "vibhakti_ids": vibhakti_ids,
        "vacana_ids": vacana_ids,
        "purusa_ids": purusa_ids,
    }


def ingest_morphology(text: str, root_vocab: Dict) -> Phase1Output:
    """
    Phase 1: Morphological Ingestion + Factorization
    
    Deterministic transformation from raw text to:
    1. Structured MorphTokens (for Phase 2A symbolic processing)
    2. Factorized ID tensors (for Phase 2B neural embeddings)
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
    
    # 4. FACTORIZE for Phase 2B (Zero OOV architecture)
    factorized = factorize_tokens(tokens, root_vocab)
    
    return {
        "tokens": tokens,
        "factorized": factorized,  # NEW: For Phase 2B
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

### Running Tests

```bash
# Run all Phase 1 tests
pytest tests/unit/test_phase1.py -v

# Run specific test class
pytest tests/unit/test_phase1.py::TestHeritageBackend -v

# Run with coverage
pytest tests/unit/test_phase1.py --cov=panini_lm.phase1_morphology
```

### Test File Location

All Phase 1 tests are in `tests/unit/test_phase1.py`.

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

---

## Concrete Input/Output Examples

### Example 1: Simple Sentence (Gita 2.47)

**Input (raw Sanskrit):**
```
कर्मणि एव ते अधिकारः
```

**Output (Phase1Output):**
```python
{
    "raw_input": "कर्मणि एव ते अधिकारः",
    "sandhi_splits": ["कर्मणि", "एव", "ते", "अधिकारः"],
    "tokens": [
        {
            "surface": "कर्मणि",
            "stem": "कर्मन्",
            "type": "subanta",
            "attributes": {
                "vibhakti": 7,      # Locative case
                "vacana": 1,        # Singular
                "linga": "n"        # Neuter
            }
        },
        {
            "surface": "एव",
            "stem": "एव",
            "type": "avyaya",       # Indeclinable particle
            "attributes": {}
        },
        {
            "surface": "ते",
            "stem": "त्वद्",
            "type": "subanta",
            "attributes": {
                "vibhakti": 6,      # Genitive case
                "vacana": 1,        # Singular
                "linga": "m"        # Masculine
            }
        },
        {
            "surface": "अधिकारः",
            "stem": "अधिकार",
            "type": "subanta",
            "attributes": {
                "vibhakti": 1,      # Nominative case
                "vacana": 1,        # Singular
                "linga": "m"        # Masculine
            }
        }
    ]
}
```

### Example 2: With Sandhi Resolution

**Input:**
```
rāmo'pi gṛhaṃ gacchati
```

**Output:**
```python
{
    "raw_input": "rāmo'pi gṛhaṃ gacchati",
    "sandhi_splits": ["rāmaḥ", "api", "gṛham", "gacchati"],  # Sandhi resolved
    "tokens": [
        {
            "surface": "rāmaḥ",
            "stem": "rāma",
            "type": "subanta",
            "attributes": {
                "vibhakti": 1,      # Nominative (subject)
                "vacana": 1,        # Singular
                "linga": "m",       # Masculine
                "karaka": "karta"   # Agent role
            }
        },
        {
            "surface": "api",
            "stem": "api",
            "type": "avyaya",
            "attributes": {}
        },
        {
            "surface": "gṛham",
            "stem": "gṛha",
            "type": "subanta",
            "attributes": {
                "vibhakti": 2,      # Accusative (object)
                "vacana": 1,
                "linga": "n",
                "karaka": "karma"   # Patient role
            }
        },
        {
            "surface": "gacchati",
            "stem": "gam",
            "type": "tinanta",      # Finite verb
            "attributes": {
                "lakara": "lat",    # Present tense
                "purusa": 1,        # Third person
                "vacana": 1         # Singular
            }
        }
    ]
}
```

### Example 3: Compound Word (Samāsa)

**Input:**
```
धर्मक्षेत्रे कुरुक्षेत्रे
```

**Output:**
```python
{
    "raw_input": "धर्मक्षेत्रे कुरुक्षेत्रे",
    "sandhi_splits": ["धर्मक्षेत्रे", "कुरुक्षेत्रे"],
    "tokens": [
        {
            "surface": "धर्मक्षेत्रे",
            "stem": "धर्मक्षेत्र",     # Compound: dharma + kṣetra
            "type": "subanta",
            "attributes": {
                "vibhakti": 7,          # Locative
                "vacana": 1,
                "linga": "n"
            }
        },
        {
            "surface": "कुरुक्षेत्रे",
            "stem": "कुरुक्षेत्र",     # Compound: kuru + kṣetra
            "type": "subanta",
            "attributes": {
                "vibhakti": 7,          # Locative
                "vacana": 1,
                "linga": "n"
            }
        }
    ]
}
```
