# Data Contracts

> Formal type definitions for all inter-phase data structures.  
> Use these types for implementation and testing validation.

---

## Table of Contents

- [Overview](#overview)
- [Phase 1 Types](#phase-1-types)
- [Phase 2A Types](#phase-2a-types)
- [Phase 2B Types](#phase-2b-types)
- [Phase 3-4 Types](#phase-3-4-types)
- [Phase 5 Types](#phase-5-types)
- [Error Types](#error-types)
- [Configuration Types](#configuration-types)
- [Validation Utilities](#validation-utilities)

---

## Overview

All data contracts are defined using Python's `typing` module with `TypedDict` for dictionary structures. These definitions serve as:

1. **Documentation**: Clear specification of expected data shapes
2. **Runtime validation**: Can be validated using `typeguard` or Pydantic
3. **IDE support**: Type hints for autocomplete and error detection
4. **Testing contracts**: Basis for test assertions

### Import Statement

```python
from typing import TypedDict, Literal, Optional, List, Dict, Union
import torch
```

---

## Phase 1 Types

### MorphAttributes

Morphological attributes extracted from a token.

```python
class MorphAttributes(TypedDict, total=False):
    """
    Grammatical attributes of a morphological token.
    All fields are optional as different token types have different attributes.
    """
    # === Nominal attributes (subanta) ===
    vibhakti: int
    """Case ending (1-8). See GLOSSARY.md for mapping."""
    
    linga: Literal["m", "f", "n"]
    """Gender: masculine, feminine, neuter."""
    
    # === Verbal attributes (tiṅanta) ===
    lakara: Optional[str]
    """Tense/mood (laṭ, liṭ, luṭ, etc.). None for non-verbs."""
    
    # === Common attributes ===
    vacana: Literal[1, 2, 3]
    """Number: 1=singular, 2=dual, 3=plural."""
    
    purusa: Literal[1, 2, 3]
    """Person: 1=3rd, 2=2nd, 3=1st (Sanskrit convention)."""
    
    # === Semantic role ===
    karaka: Optional[Literal["karta", "karma", "karana", "sampradana", "apadana", "adhikarana"]]
    """Semantic role. May be inferred in Phase 2A."""
```

### MorphToken

A single morphologically analyzed token.

```python
class MorphToken(TypedDict):
    """
    Output of Phase 1 morphological analysis for a single word.
    """
    surface: str
    """Surface form as it appears after sandhi resolution. Example: 'rāmaḥ'"""
    
    stem: str
    """Base form (prakṛti). Example: 'rāma' (nominal) or 'gam' (verbal root)"""
    
    type: Literal["subanta", "tinanta", "avyaya", "krdanta"]
    """
    Token category:
    - subanta: nominal (noun/adjective/pronoun)
    - tinanta: finite verb
    - avyaya: indeclinable (particle/conjunction)
    - krdanta: verbal derivative (participle/infinitive)
    """
    
    attributes: MorphAttributes
    """Grammatical attributes. Contents vary by token type."""
```

### Phase1Output

Complete output of Phase 1.

```python
class Phase1Output(TypedDict):
    """
    Complete output of Phase 1 Morphological Ingestion.
    """
    tokens: List[MorphToken]
    """Ordered list of analyzed tokens."""
    
    raw_input: str
    """Original input string for debugging."""
    
    sandhi_splits: List[str]
    """Intermediate sandhi-resolved forms before full analysis."""
```

### Example

```python
# Input: "rāmo'pi gṛhaṃ gacchati"
# Output:
phase1_output: Phase1Output = {
    "raw_input": "rāmo'pi gṛhaṃ gacchati",
    "sandhi_splits": ["rāmaḥ", "api", "gṛham", "gacchati"],
    "tokens": [
        {
            "surface": "rāmaḥ",
            "stem": "rāma",
            "type": "subanta",
            "attributes": {
                "vibhakti": 1,
                "vacana": 1,
                "linga": "m",
                "karaka": "karta"
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
                "vibhakti": 2,
                "vacana": 1,
                "linga": "n",
                "karaka": "karma"
            }
        },
        {
            "surface": "gacchati",
            "stem": "gam",
            "type": "tinanta",
            "attributes": {
                "purusa": 1,
                "vacana": 1,
                "lakara": "lat"
            }
        }
    ]
}
```

---

## Phase 2A Types

### AdjacencyMatrix

Sparse grammatical routing matrix.

```python
class AdjacencyMatrixMeta(TypedDict):
    """
    Metadata for the adjacency matrix.
    """
    seq_len: int
    """Sequence length N."""
    
    num_valid_edges: int
    """Count of valid (non -∞) edges."""
    
    sparsity_ratio: float
    """Ratio of valid edges to total: num_valid_edges / (N * N)."""
    
    avg_connections_per_token: float
    """Average k value: num_valid_edges / N."""


class AdjacencyMatrix(TypedDict):
    """
    Phase 2A output: grammatical adjacency matrix M.
    """
    matrix: "torch.Tensor"
    """
    Shape: (N, N), dtype: torch.float32
    Values:
    - 0.0: Valid grammatical connection (token i may attend to j)
    - float('-inf'): Invalid connection (will zero out after softmax)
    """
    
    meta: AdjacencyMatrixMeta
    """Statistical metadata for validation and debugging."""
```

### GrammaticalLink

Individual grammatical relationship.

```python
class GrammaticalLink(TypedDict):
    """
    Single grammatical link identified by Phase 2A.
    Used for debugging and rule tracing.
    """
    source_idx: int
    """Index of source token."""
    
    target_idx: int
    """Index of target token."""
    
    link_type: str
    """Type of relationship: 'subject-verb', 'object-verb', 'modifier', etc."""
    
    rule_applied: Optional[str]
    """Sūtra or rule that validated this link."""
```

### Phase2AOutput

```python
class Phase2AOutput(TypedDict):
    """
    Complete output of Phase 2A Symbolic Engine.
    """
    adjacency_matrix: AdjacencyMatrix
    """The sparse matrix M."""
    
    links: List[GrammaticalLink]
    """Detailed link information for debugging."""
```

### Example

```python
import torch

# For sequence of 4 tokens: ["rāmaḥ", "api", "gṛham", "gacchati"]
phase2a_output: Phase2AOutput = {
    "adjacency_matrix": {
        "matrix": torch.tensor([
            [0.0,    float('-inf'), float('-inf'), 0.0],    # rāmaḥ → gacchati
            [float('-inf'), 0.0,    float('-inf'), float('-inf')],  # api (particle)
            [float('-inf'), float('-inf'), 0.0,    0.0],    # gṛham → gacchati
            [0.0,    float('-inf'), 0.0,    0.0]            # verb self + relations
        ], dtype=torch.float32),
        "meta": {
            "seq_len": 4,
            "num_valid_edges": 7,
            "sparsity_ratio": 0.4375,
            "avg_connections_per_token": 1.75
        }
    },
    "links": [
        {"source_idx": 0, "target_idx": 3, "link_type": "subject-verb", "rule_applied": "kartā-kriyā"},
        {"source_idx": 2, "target_idx": 3, "link_type": "object-verb", "rule_applied": "karma-kriyā"}
    ]
}
```

---

## Phase 2B Types

### Vocabulary Configuration (~4,000 primitives)

Because Phase 1 decomposes words into pure mathematical primitives, the vocabulary is strictly limited:

| Category | Count | Description |
|----------|-------|-------------|
| **Dhātus** (Verbal roots) | ~2,000 | √gam, √bhū, √kṛ, √as, etc. |
| **Upasargas** (Prefixes) | ~20 | pra-, upa-, sam-, vi-, etc. |
| **Pratyayas** (Core affixes) | ~100-200 | -ta, -tavya, -ya, etc. |
| **Prātipadikas** (Nominal stems) | ~1,500 | rāma-, gṛha-, deva-, etc. |
| **Special tokens** | ~10 | [PAD], [UNK], [BOS], [EOS], [MASK] |
| **Total** | **~4,000** | vs. 50,000+ in standard LLMs |

### VocabMapping

```python
class VocabMapping(TypedDict):
    """
    Vocabulary mapping for factorized tokenization.
    
    Unlike standard LLMs with 50k+ surface forms, Panini-LM maps only
    morphological primitives: roots, stems, and special tokens.
    """
    root_to_id: Dict[str, int]
    """Mapping from root/stem string to integer ID (~4000 entries)."""
    
    id_to_root: Dict[int, str]
    """Reverse mapping for decoding."""
    
    special_tokens: Dict[str, int]
    """Special tokens: [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3, [MASK]=4."""
    
    vocab_size: int
    """Total vocabulary size (~4000)."""
```

### Grammatical ID Mappings

```python
# Token type encoding
TYPE_TO_ID = {
    "subanta": 0,    # Nominal (noun/adjective/pronoun)
    "tinanta": 1,    # Finite verb
    "avyaya": 2,     # Indeclinable (particle/conjunction)
    "krdanta": 3,    # Verbal derivative (participle/infinitive)
    "taddhita": 4,   # Secondary derivative
    "samasa": 5,     # Compound
    "none": 6,       # Special tokens
}

# Case (vibhakti) encoding
VIBHAKTI_TO_ID = {
    "none": 0,       # Non-nominals or special tokens
    1: 1,            # Prathamā (nominative)
    2: 2,            # Dvitīyā (accusative)
    3: 3,            # Tṛtīyā (instrumental)
    4: 4,            # Caturthī (dative)
    5: 5,            # Pañcamī (ablative)
    6: 6,            # Ṣaṣṭhī (genitive)
    7: 7,            # Saptamī (locative)
    "vocative": 8,   # Sambodhana (vocative)
}

# Number (vacana) encoding
VACANA_TO_ID = {
    "none": 0,       # Non-applicable
    1: 1,            # Ekavacana (singular)
    2: 2,            # Dvivacana (dual)
    3: 3,            # Bahuvacana (plural)
}

# Person (puruṣa) encoding — Sanskrit convention
PURUSA_TO_ID = {
    "none": 0,       # Non-verbs
    1: 1,            # Prathama-puruṣa (3rd person)
    2: 2,            # Madhyama-puruṣa (2nd person)
    3: 3,            # Uttama-puruṣa (1st person)
}
```

### FactorizedTokenBatch

**THE CORE DATA STRUCTURE** — Input to Phase 2B Neural Engine.

```python
class FactorizedTokenBatch(TypedDict):
    """
    Factorized token representation for Panini-LM.
    
    Instead of a single `token_ids` array, we provide parallel tensors
    for each morphological dimension. The embedding layer sums these
    to construct the final word embedding.
    
    This enables:
    1. Zero OOV errors for any valid inflection
    2. 12× parameter reduction in embeddings
    3. Structural encoding of morphological knowledge
    """
    root_ids: "torch.LongTensor"
    """
    Root/stem IDs — THE SEMANTIC CORE.
    Shape: (batch_size, seq_len), dtype: torch.long
    Values: 0-3999 (from ~4000 primitive vocabulary)
    """
    
    type_ids: "torch.LongTensor"
    """
    Token type IDs.
    Shape: (batch_size, seq_len), dtype: torch.long
    Values: 0=subanta, 1=tiṅanta, 2=avyaya, 3=kṛdanta, 4=taddhita, 5=samāsa, 6=none
    """
    
    vibhakti_ids: "torch.LongTensor"
    """
    Case (vibhakti) IDs — for nominals.
    Shape: (batch_size, seq_len), dtype: torch.long
    Values: 0=none, 1-7=cases, 8=vocative
    """
    
    vacana_ids: "torch.LongTensor"
    """
    Number (vacana) IDs.
    Shape: (batch_size, seq_len), dtype: torch.long
    Values: 0=none, 1=singular, 2=dual, 3=plural
    """
    
    purusa_ids: "torch.LongTensor"
    """
    Person (puruṣa) IDs — for verbs.
    Shape: (batch_size, seq_len), dtype: torch.long
    Values: 0=none, 1=3rd (prathama), 2=2nd (madhyama), 3=1st (uttama)
    """
    
    attention_mask: "torch.BoolTensor"
    """Shape: (batch_size, seq_len), True for valid positions."""
    
    seq_lengths: List[int]
    """Actual sequence lengths before padding."""
```

### Example: Encoding "rāmo gṛhaṃ gacchati"

```python
# After Phase 1 morphological analysis:
# rāmaḥ  → stem="rāma", type=subanta, vibhakti=1, vacana=1
# gṛham  → stem="gṛha", type=subanta, vibhakti=2, vacana=1
# gacchati → stem="gam", type=tiṅanta, puruṣa=1, vacana=1

factorized_batch: FactorizedTokenBatch = {
    "root_ids": torch.tensor([[2, 1130, 847, 502, 3]]),  # [BOS], rāma, gṛha, gam, [EOS]
    "type_ids": torch.tensor([[6, 0, 0, 1, 6]]),         # none, subanta, subanta, tiṅanta, none
    "vibhakti_ids": torch.tensor([[0, 1, 2, 0, 0]]),     # none, nom, acc, none, none
    "vacana_ids": torch.tensor([[0, 1, 1, 1, 0]]),       # none, sing, sing, sing, none
    "purusa_ids": torch.tensor([[0, 0, 0, 1, 0]]),       # none, none, none, 3rd, none
    "attention_mask": torch.tensor([[True, True, True, True, True]]),
    "seq_lengths": [5],
}
```

### QKVTensors

Query, Key, Value projections.

```python
class QKVTensors(TypedDict):
    """
    Phase 2B output: Q, K, V tensors for attention.
    """
    Q: "torch.Tensor"
    """Query tensor. Shape: (batch_size, num_heads, seq_len, head_dim)"""
    
    K: "torch.Tensor"
    """Key tensor. Shape: (batch_size, num_heads, seq_len, head_dim)"""
    
    V: "torch.Tensor"
    """Value tensor. Shape: (batch_size, num_heads, seq_len, head_dim)"""
```

### Phase2BOutput

```python
class Phase2BOutput(TypedDict):
    """
    Complete output of Phase 2B Neural Engine.
    """
    embeddings: "torch.Tensor"
    """
    Factorized embeddings — sum of morphological components.
    Shape: (batch_size, seq_len, d_model)
    
    E(token) = E(root) + E(type) + E(vibhakti) + E(vacana) + E(puruṣa)
    """
    
    qkv: QKVTensors
    """Projected Q, K, V tensors."""
```

---

## Phase 3-4 Types

### AttentionOutput

```python
class AttentionOutput(TypedDict):
    """
    Output of Phase 3 Sparse Attention.
    """
    hidden_states: "torch.Tensor"
    """Contextualized representations. Shape: (batch_size, seq_len, d_model)"""
    
    attention_weights: Optional["torch.Tensor"]
    """
    Optional attention weights for visualization.
    Shape: (batch_size, num_heads, seq_len, seq_len)
    Only populated if return_attention=True.
    """


class FFNOutput(TypedDict):
    """
    Output of Phase 4 Semantic Maturation.
    """
    hidden_states: "torch.Tensor"
    """Refined representations. Shape: (batch_size, seq_len, d_model)"""
```

---

## Phase 5 Types

### MorphologicalState

Decoding state tracker.

```python
class MorphologicalState(TypedDict):
    """
    State tracking for grammar-constrained decoding.
    """
    last_token: Optional[MorphToken]
    """Most recently generated token."""
    
    pending_agreements: List[Dict]
    """Unresolved agreement constraints (e.g., subject awaiting verb)."""
    
    open_clauses: List[Dict]
    """Incomplete syntactic structures."""
    
    generated_tokens: List[MorphToken]
    """All tokens generated so far."""
```

### GrammarMask

Grammar constraint mask.

```python
class GrammarMask(TypedDict):
    """
    Mask for grammar-constrained decoding.
    """
    mask: "torch.Tensor"
    """
    Shape: (vocab_size,), dtype: torch.float32
    Values:
    - 0.0: Token is grammatically valid
    - float('-inf'): Token is grammatically impossible
    """
    
    valid_token_count: int
    """Number of valid (non-masked) tokens."""
    
    valid_token_ids: List[int]
    """IDs of valid tokens (for debugging)."""
```

### DecodingOutput

```python
class DecodingOutput(TypedDict):
    """
    Output of Phase 5 Grammar-Constrained Decoding.
    """
    generated_text: str
    """Final generated text."""
    
    generated_tokens: List[MorphToken]
    """Token-level breakdown."""
    
    logits_history: Optional[List["torch.Tensor"]]
    """Per-step logits (if requested)."""
    
    grammar_violations: int
    """Should always be 0 with proper masking."""
```

---

## Error Types

### PaniniLMError (Base)

```python
class PaniniLMError(Exception):
    """Base exception for all Panini-LM errors."""
    pass
```

### Phase 1 Errors

```python
class MorphologyError(PaniniLMError):
    """Errors during morphological analysis."""
    pass


class SandhiResolutionError(MorphologyError):
    """Failed to resolve sandhi in input text."""
    
    def __init__(self, input_text: str, position: int, message: str):
        self.input_text = input_text
        self.position = position
        super().__init__(f"Sandhi resolution failed at position {position}: {message}")


class UnknownTokenError(MorphologyError):
    """Token not found in morphological database."""
    
    def __init__(self, token: str):
        self.token = token
        super().__init__(f"Unknown token: '{token}'")
```

### Phase 2A Errors

```python
class SymbolicEngineError(PaniniLMError):
    """Errors in the symbolic engine."""
    pass


class RuleConflictError(SymbolicEngineError):
    """Multiple conflicting rules apply."""
    
    def __init__(self, token_i: int, token_j: int, rules: List[str]):
        self.token_i = token_i
        self.token_j = token_j
        self.rules = rules
        super().__init__(f"Rule conflict for ({token_i}, {token_j}): {rules}")


class InvalidGrammarError(SymbolicEngineError):
    """Input violates grammatical constraints."""
    pass
```

### Phase 3 Errors

```python
class AttentionError(PaniniLMError):
    """Errors in attention computation."""
    pass


class KernelError(AttentionError):
    """Triton/CUDA kernel error."""
    
    def __init__(self, kernel_name: str, message: str):
        self.kernel_name = kernel_name
        super().__init__(f"Kernel '{kernel_name}' failed: {message}")
```

### Phase 5 Errors

```python
class DecodingError(PaniniLMError):
    """Errors during decoding."""
    pass


class NoValidTokensError(DecodingError):
    """Grammar constraints eliminate all possible next tokens."""
    
    def __init__(self, state: MorphologicalState):
        self.state = state
        super().__init__("No grammatically valid next tokens available")
```

---

## Configuration Types

### ModelConfig

```python
class ModelConfig(TypedDict):
    """
    Model hyperparameters.
    """
    # Dimensions
    d_model: int
    """Hidden dimension. Default: 512"""
    
    num_heads: int
    """Number of attention heads. Default: 8"""
    
    num_layers: int
    """Number of transformer layers. Default: 6"""
    
    vocab_size: int
    """Vocabulary size. ~4,000 morphological primitives (factorized embeddings)."""
    
    # FFN
    ffn_expansion: float
    """FFN expansion factor. Default: 2.0 (reduced from standard 4.0)"""
    
    # Attention
    use_sparse_attention: bool
    """Whether to use Phase 3 sparse attention. Default: True"""
    
    # Decoding
    use_grammar_constraints: bool
    """Whether to use Phase 5 grammar masking. Default: True"""
```

### TrainingConfig

```python
class TrainingConfig(TypedDict):
    """
    Training hyperparameters.
    """
    batch_size: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    gradient_accumulation: int
    weight_decay: float
    max_seq_len: int
```

---

## Validation Utilities

### Type Validators

```python
def validate_morph_token(token: dict) -> MorphToken:
    """
    Validate and cast a dictionary to MorphToken.
    Raises TypeError if validation fails.
    """
    required_keys = {"surface", "stem", "type", "attributes"}
    if not required_keys.issubset(token.keys()):
        missing = required_keys - token.keys()
        raise TypeError(f"MorphToken missing keys: {missing}")
    
    valid_types = {"subanta", "tinanta", "avyaya", "krdanta"}
    if token["type"] not in valid_types:
        raise TypeError(f"Invalid token type: {token['type']}")
    
    return token  # type: ignore


def validate_adjacency_matrix(matrix: torch.Tensor, seq_len: int) -> None:
    """
    Validate adjacency matrix shape and values.
    """
    if matrix.shape != (seq_len, seq_len):
        raise ValueError(f"Expected shape ({seq_len}, {seq_len}), got {matrix.shape}")
    
    if matrix.dtype != torch.float32:
        raise TypeError(f"Expected float32, got {matrix.dtype}")
    
    # Check values are either 0.0 or -inf
    valid_values = (matrix == 0.0) | (matrix == float('-inf'))
    if not valid_values.all():
        invalid_count = (~valid_values).sum().item()
        raise ValueError(f"Matrix contains {invalid_count} invalid values (not 0.0 or -inf)")


def validate_sparsity(matrix: torch.Tensor, max_avg_k: float = 5.0) -> float:
    """
    Validate matrix sparsity is within expected bounds.
    Returns average connections per token.
    """
    seq_len = matrix.shape[0]
    valid_edges = (matrix == 0.0).sum().item()
    avg_k = valid_edges / seq_len
    
    if avg_k > max_avg_k:
        raise ValueError(f"Matrix too dense: avg_k={avg_k:.2f} > {max_avg_k}")
    
    return avg_k
```

---

## JSON Schema (OpenAPI)

For REST API integration, equivalent JSON Schema definitions:

```json
{
  "MorphToken": {
    "type": "object",
    "required": ["surface", "stem", "type", "attributes"],
    "properties": {
      "surface": {"type": "string"},
      "stem": {"type": "string"},
      "type": {
        "type": "string",
        "enum": ["subanta", "tinanta", "avyaya", "krdanta"]
      },
      "attributes": {"$ref": "#/MorphAttributes"}
    }
  },
  "MorphAttributes": {
    "type": "object",
    "properties": {
      "vibhakti": {"type": "integer", "minimum": 1, "maximum": 8},
      "vacana": {"type": "integer", "enum": [1, 2, 3]},
      "purusa": {"type": "integer", "enum": [1, 2, 3]},
      "linga": {"type": "string", "enum": ["m", "f", "n"]},
      "lakara": {"type": "string", "nullable": true},
      "karaka": {
        "type": "string",
        "enum": ["karta", "karma", "karana", "sampradana", "apadana", "adhikarana"],
        "nullable": true
      }
    }
  }
}
```

---

## See Also

- [Phase 1 — Morphology](../phases/phase1-morphology.md)
- [Phase 2A — Symbolic Engine](../phases/phase2a-symbolic.md)
- [Phase 5 — Decoding](../phases/phase5-decoding.md)
- [Test Specifications](../testing/test-specifications.md)
- [Glossary](../GLOSSARY.md)
