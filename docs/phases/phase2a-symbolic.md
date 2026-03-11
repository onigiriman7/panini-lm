# Phase 2A — Symbolic Engine

> Generate the deterministic grammatical adjacency matrix M.

---

## Overview

Phase 2A processes morphological tokens from Phase 1 to produce a sparse **Adjacency Matrix M** that encodes which token pairs can validly attend to each other based on Pāṇinian grammar rules.

This is the **Symbolic Track** — purely deterministic, no learned parameters.

---

## Input/Output Contract

### Input

- **Type**: `List[MorphToken]` from Phase 1
- **Requirements**: Each token must have `type` and `attributes`

### Output

- **Type**: `Phase2AOutput` (see [data-contracts.md](../types/data-contracts.md))
- **Contains**: Adjacency matrix M and link metadata

```python
{
    "adjacency_matrix": {
        "matrix": tensor([[0.0, -inf, 0.0], [-inf, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "meta": {
            "seq_len": 3,
            "num_valid_edges": 6,
            "sparsity_ratio": 0.67,
            "avg_connections_per_token": 2.0
        }
    },
    "links": [
        {"source_idx": 0, "target_idx": 2, "link_type": "subject-verb", "rule_applied": "kartā-kriyā"}
    ]
}
```

### Matrix Values

| Value | Meaning | Effect in Attention |
|-------|---------|---------------------|
| `0.0` | Valid grammatical connection | Normal attention score |
| `-∞` | Invalid connection | After softmax → 0 probability |

### Errors

- `RuleConflictError`: Multiple conflicting rules for a token pair
- `InvalidGrammarError`: Input violates fundamental grammatical constraints

---

## Dependencies

- **Input**: Phase 1 output (`List[MorphToken]`)
- **External**: [samsadhani](../integration/samsadhani.md) (optional, for training data)
- **Output consumers**: Phase 3 (Sparse Attention)

---

## Implementation Details

### Rule Categories

The symbolic engine evaluates several categories of grammatical relationships:

#### 1. Subject-Verb Agreement (Kartā-Kriyā)

```python
def check_subject_verb(head: MorphToken, dep: MorphToken) -> bool:
    """Check nominative noun can link to verb with matching agreement."""
    if head["type"] != "subanta" or dep["type"] != "tinanta":
        return False
    
    # Nominative case (vibhakti 1) required for subject
    if head["attributes"].get("vibhakti") != 1:
        return False
    
    # Number agreement (vacana)
    return head["attributes"].get("vacana") == dep["attributes"].get("vacana")
```

#### 2. Object-Verb (Karma-Kriyā)

```python
def check_object_verb(head: MorphToken, dep: MorphToken) -> bool:
    """Check accusative noun can link to verb."""
    if head["type"] != "subanta" or dep["type"] != "tinanta":
        return False
    
    # Accusative case (vibhakti 2) for direct object
    return head["attributes"].get("vibhakti") == 2
```

#### 3. Self-Attention

```python
def check_self(head: MorphToken, dep: MorphToken) -> bool:
    """Self-attention always allowed."""
    return head is dep
```

### Matrix Construction

```python
def build_matrix_M(tokens: List[MorphToken]) -> Phase2AOutput:
    """
    Phase 2A: Construct adjacency matrix M.
    
    Complexity: O(N²) — evaluates each token pair.
    Deterministic: Same input always produces same output.
    """
    N = len(tokens)
    M = torch.full((N, N), float('-inf'), dtype=torch.float32)
    links = []
    
    rules = [
        (check_subject_verb, "kartā-kriyā"),
        (check_object_verb, "karma-kriyā"),
        (check_instrument_verb, "karaṇa-kriyā"),
        (check_self, "sva-sambandha"),
        # ... more rules
    ]
    
    for i, t_i in enumerate(tokens):
        for j, t_j in enumerate(tokens):
            for rule_fn, rule_name in rules:
                if rule_fn(t_i, t_j):
                    M[i, j] = 0.0
                    links.append({
                        "source_idx": i,
                        "target_idx": j,
                        "link_type": infer_type(t_i, t_j),
                        "rule_applied": rule_name
                    })
                    break  # First matching rule wins
    
    return {
        "adjacency_matrix": {
            "matrix": M,
            "meta": compute_meta(M)
        },
        "links": links
    }
```

---

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| `RuleConflictError` | Multiple rules return different results | Apply priority ordering |
| `InvalidGrammarError` | Sequence has no valid parse | Return maximally sparse M, log warning |
| Zero valid edges | Completely disconnected | Allow self-attention only |

### Rule Priority

When multiple rules could apply, use this priority:

1. Self-attention (always allowed)
2. Verb-argument relations (kartā, karma, etc.)
3. Modifier relations
4. Default: no connection

---

## Test Specifications

### Unit Tests

```python
def test_subject_verb_link():
    """Subject (nominative) should link to verb."""
    tokens = [
        {"type": "subanta", "attributes": {"vibhakti": 1, "vacana": 1}},
        {"type": "tinanta", "attributes": {"vacana": 1}}
    ]
    result = build_matrix_M(tokens)
    assert result["adjacency_matrix"]["matrix"][0, 1] == 0.0

def test_number_mismatch():
    """Subject-verb with mismatched number should not link."""
    tokens = [
        {"type": "subanta", "attributes": {"vibhakti": 1, "vacana": 1}},  # singular
        {"type": "tinanta", "attributes": {"vacana": 3}}  # plural
    ]
    result = build_matrix_M(tokens)
    assert result["adjacency_matrix"]["matrix"][0, 1] == float('-inf')

def test_sparsity():
    """Matrix should be sparse (avg k ≈ 2-3)."""
    tokens = generate_typical_sentence(length=20)
    result = build_matrix_M(tokens)
    avg_k = result["adjacency_matrix"]["meta"]["avg_connections_per_token"]
    assert 1.0 <= avg_k <= 5.0
```

### Validation Corpus

```python
VALIDATION_CASES = [
    {
        "sentence": "rāmaḥ gacchati",
        "expected_edges": [(0, 1), (0, 0), (1, 1)],  # subject→verb, self
    },
    {
        "sentence": "rāmaḥ gṛham gacchati",
        "expected_edges": [(0, 2), (1, 2)],  # subject→verb, object→verb
    }
]
```

---

## Related Documents

- [Data Contracts](../types/data-contracts.md) — `AdjacencyMatrix`, `GrammaticalLink` definitions
- [samsadhani Integration](../integration/samsadhani.md) — External Kāraka API
- [Glossary](../GLOSSARY.md) — Kāraka, Vibhakti definitions
- [Phase 1](phase1-morphology.md) — Input source
- [Phase 3](phase3-attention.md) — Uses Matrix M for sparse routing

---

## Testing Guide

### Running Tests

```bash
# Run all Phase 2A tests
pytest tests/unit/test_phase2a.py -v

# Run with coverage
pytest tests/unit/test_phase2a.py --cov=panini_lm.phase2a_symbolic

# Run specific test class
pytest tests/unit/test_phase2a.py::TestSubjectVerbRule -v
```

### Test Categories

| Category | Description | Test Class |
|----------|-------------|------------|
| Rule tests | Individual grammar rule behavior | `TestSubjectVerbRule`, `TestObjectVerbRule`, etc. |
| Matrix tests | Adjacency matrix construction | `TestBuildAdjacencyMatrix` |
| Sparsity tests | Verify expected sparsity levels | `TestSparsity` |
| Metadata tests | Meta computation correctness | `TestComputeAdjacencyMeta` |

### Usage Examples

```python
from panini_lm.phase2a_symbolic import build_adjacency_matrix, get_default_rules

# Build adjacency matrix from tokens
tokens = [
    {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta", 
     "attributes": {"vibhakti": 1, "vacana": 1}},
    {"surface": "gacchati", "stem": "gam", "type": "tinanta",
     "attributes": {"vacana": 1}}
]

adj = build_adjacency_matrix(tokens)

# Check matrix shape
print(f"Matrix shape: {adj.matrix.shape}")  # (2, 2)

# Check valid edges
print(f"Valid edges: {adj.meta['num_valid_edges']}")

# Check sparsity
print(f"Sparsity: {adj.meta['sparsity_ratio']:.2%}")

# View links
for link in adj.links:
    print(f"{link['source_idx']} → {link['target_idx']}: {link['rule_applied']}")

# Custom rules
from panini_lm.phase2a_symbolic import SubjectVerbRule, ObjectVerbRule

custom_rules = [SubjectVerbRule(), ObjectVerbRule()]
adj_custom = build_adjacency_matrix(tokens, rules=custom_rules)
```

### Debugging Matrix

```python
from panini_lm.phase2a_symbolic.matrix_builder import visualize_matrix

# ASCII visualization
print(visualize_matrix(adj, tokens))
```

### Key Assertions

```python
# Subject-verb link should exist
assert adj.matrix[0, 1] == 0.0

# Self-attention always exists
assert adj.matrix[0, 0] == 0.0

# Average connections per token (target: 2-5)
assert 1.0 <= adj.meta["avg_connections_per_token"] <= 5.0
```
