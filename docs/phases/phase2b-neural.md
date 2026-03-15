# Phase 2B — Neural Engine

> **Factorized Embeddings**: Construct word meaning dynamically from morphological primitives.

---

## Overview

Phase 2B is the **Neural Track** — it transforms morphological tokens into dense semantic representations using **Vectorized Factorization**, the key breakthrough that enables radical parameter reduction and zero OOV errors for inflected forms.

### Critical Design Principles

1. **No Positional Encoding** (RoPE, sinusoidal, etc.) — Sanskrit has free word order. Position should not bias attention.
2. **Factorized Embeddings** — The model does NOT embed fully inflected words (*padas*). Instead, it **constructs** each word's embedding by summing embeddings for its morphological primitives (root + grammatical tags).

### Why Factorized Embeddings?

Traditional Transformers embed each surface form separately:
- "gacchati", "gacchāmi", "gacchasi", "gacchataḥ"... → 50,000+ separate vectors

Panini-LM decomposes words into mathematical primitives:
- "gacchati" = √gam (root) + tiṅanta (type) + laṭ (tense) + prathama-puruṣa + eka-vacana
- All forms of √gam share the SAME root embedding — the model learns morphological compositionality.

**Result**: Zero OOV errors for any valid Sanskrit inflection. The model can embed forms it has never seen, as long as it knows the root and the grammatical tags.

---

## Vocabulary Size: ~4,000 Primitives

Because Phase 1 decomposes words into pure mathematical primitives, the vocabulary is strictly limited:

| Category | Count | Description |
|----------|-------|-------------|
| **Dhātus** (Verbal roots) | ~2,000 | √gam, √bhū, √kṛ, √as, etc. |
| **Upasargas** (Prefixes) | ~20 | pra-, upa-, sam-, vi-, etc. |
| **Pratyayas** (Core affixes) | ~100-200 | -ta, -tavya, -ya, etc. |
| **Prātipadikas** (Nominal stems) | ~1,500 | rāma-, gṛha-, deva-, etc. |
| **Special tokens** | ~10 | [PAD], [UNK], [BOS], [EOS], [MASK] |
| **Total** | **~4,000** | vs. 50,000+ in standard LLMs |

This is the **Token Compression Breakthrough** — a 12× reduction in vocabulary size.

---

## Input/Output Contract

### Input

- **Type**: `FactorizedTokenBatch` from Phase 1 (see [data-contracts.md](../types/data-contracts.md))
- **Fields**: `root_ids`, `type_ids`, `vibhakti_ids`, `vacana_ids`, `purusa_ids`

### Output

- **Type**: `Phase2BOutput`
- **Contains**: Factorized embeddings and Q, K, V tensors

```python
{
    "embeddings": tensor(...),  # (batch, seq, d_model)
    "qkv": {
        "Q": tensor(...),  # (batch, heads, seq, head_dim)
        "K": tensor(...),  # (batch, heads, seq, head_dim)
        "V": tensor(...)   # (batch, heads, seq, head_dim)
    }
}
```

### Tensor Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `embeddings` | (B, N, D) | Factorized token embeddings |
| `Q`, `K`, `V` | (B, H, N, D/H) | Multi-head projections |

Where B=batch, N=seq_len, D=d_model, H=num_heads.

---

## Dependencies

- **Input**: Phase 1 output (`FactorizedTokenBatch`)
- **External**: PyTorch (`torch.nn`)
- **Output consumers**: Phase 3 (Sparse Attention)

---

## Implementation Details

### Factorized Tokenization

Convert `MorphToken` objects into parallel ID tensors:

```python
class FactorizedTokenizer:
    """
    Tokenizer that produces parallel ID tensors for factorized embedding.
    
    Unlike standard tokenizers that map each surface form to one ID,
    this produces multiple ID tensors — one for each morphological dimension.
    """
    
    def __init__(self, vocab_path: str):
        self.root_to_id = load_vocab(vocab_path, "roots")  # ~4000 roots/stems
        self.pad_id = 0
        self.unk_id = 1
        
        # ID mappings for grammatical dimensions
        self.type_to_id = {
            "subanta": 0, "tinanta": 1, "avyaya": 2, "krdanta": 3,
            "taddhita": 4, "samasa": 5, "none": 6
        }
        self.vibhakti_to_id = {
            "none": 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, "vocative": 8
        }
        self.vacana_to_id = {"none": 0, 1: 1, 2: 2, 3: 3}  # sing, dual, plural
        self.purusa_to_id = {"none": 0, 1: 1, 2: 2, 3: 3}  # 3rd, 2nd, 1st
    
    def encode(self, tokens: List[MorphToken]) -> FactorizedTokenBatch:
        """Convert MorphTokens to parallel ID tensors."""
        root_ids, type_ids, vibhakti_ids, vacana_ids, purusa_ids = [], [], [], [], []
        
        for token in tokens:
            # Root/stem is the semantic core
            root_ids.append(self.root_to_id.get(token["stem"], self.unk_id))
            
            # Grammatical dimensions
            type_ids.append(self.type_to_id.get(token["type"], 6))
            attrs = token.get("attributes", {})
            vibhakti_ids.append(self.vibhakti_to_id.get(attrs.get("vibhakti"), 0))
            vacana_ids.append(self.vacana_to_id.get(attrs.get("vacana"), 0))
            purusa_ids.append(self.purusa_to_id.get(attrs.get("purusa"), 0))
        
        return {
            "root_ids": torch.tensor(root_ids, dtype=torch.long),
            "type_ids": torch.tensor(type_ids, dtype=torch.long),
            "vibhakti_ids": torch.tensor(vibhakti_ids, dtype=torch.long),
            "vacana_ids": torch.tensor(vacana_ids, dtype=torch.long),
            "purusa_ids": torch.tensor(purusa_ids, dtype=torch.long),
        }
```

### Factorized Embedding Layer (Core Innovation)

```python
class PaninianEmbedding(nn.Module):
    """
    Factorized Embedding Layer — THE KEY TO PANINI-LM.
    
    Traditional Transformers: nn.Embedding(50000, d_model) → 25.6M parameters
    Panini-LM: Additive composition of small embedding matrices → 2.1M parameters
    
    The neural network sees each word as the SUM of its mathematical parts:
        E(gacchati) = E(√gam) + E(tiṅanta) + E(laṭ) + E(prathama) + E(eka)
    
    Benefits:
    1. ZERO OOV for inflections — all valid forms are compositionally derivable
    2. 12× parameter reduction in embeddings
    3. Morphological knowledge is encoded structurally, not learned implicitly
    """
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        
        # === Core Semantic Embeddings (Roots & Stems) ===
        # ~4000 items: dhātus, prātipadikas, upasargas, pratyayas, special tokens
        self.root_embed = nn.Embedding(4000, d_model)
        
        # === Grammatical Meta-Data Embeddings (Tiny Matrices) ===
        self.type_embed = nn.Embedding(7, d_model)      # subanta, tiṅanta, avyaya, kṛdanta, taddhita, samāsa, none
        self.vibhakti_embed = nn.Embedding(9, d_model)  # 1-7 + vocative + none
        self.vacana_embed = nn.Embedding(4, d_model)    # singular, dual, plural, none
        self.purusa_embed = nn.Embedding(4, d_model)    # 3rd, 2nd, 1st, none (Sanskrit convention)
        
    def forward(
        self,
        root_ids: torch.LongTensor,
        type_ids: torch.LongTensor,
        vibhakti_ids: torch.LongTensor,
        vacana_ids: torch.LongTensor,
        purusa_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Construct embeddings by summing morphological components.
        
        Args:
            root_ids: (batch, seq) — Root/stem IDs (the semantic core)
            type_ids: (batch, seq) — Token type (subanta/tiṅanta/etc.)
            vibhakti_ids: (batch, seq) — Case (1-7, vocative, or none)
            vacana_ids: (batch, seq) — Number (singular/dual/plural/none)
            purusa_ids: (batch, seq) — Person (3rd/2nd/1st/none)
        
        Returns:
            (batch, seq, d_model) — Factorized embeddings
        """
        # The neural network sees the word as a sum of its mathematical parts.
        return (
            self.root_embed(root_ids) +
            self.type_embed(type_ids) +
            self.vibhakti_embed(vibhakti_ids) +
            self.vacana_embed(vacana_ids) +
            self.purusa_embed(purusa_ids)
        )
```

### Parameter Count Comparison

| Component | Standard Transformer | Panini-LM |
|-----------|---------------------|-----------|
| Token Embedding | 50,000 × 512 = **25,600,000** | — |
| Root Embedding | — | 4,000 × 512 = **2,048,000** |
| Type Embedding | — | 7 × 512 = **3,584** |
| Vibhakti Embedding | — | 9 × 512 = **4,608** |
| Vacana Embedding | — | 4 × 512 = **2,048** |
| Purusa Embedding | — | 4 × 512 = **2,048** |
| **Total Embedding** | **25,600,000** | **2,060,288** |

**Reduction: 12.4× fewer parameters in the embedding layer alone.**

### Q/K/V Projection

```python
class QKVProjection(nn.Module):
    """Project factorized embeddings to Query, Key, Value tensors."""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> QKVTensors:
        """
        Args:
            x: (batch, seq, d_model) — Factorized embeddings
        Returns:
            Q, K, V each of shape (batch, heads, seq, head_dim)
        """
        batch, seq, _ = x.shape
        
        Q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        return {"Q": Q, "K": K, "V": V}
```

### Complete Phase 2B — Neural Engine

```python
class NeuralEngine(nn.Module):
    """
    Phase 2B: Neural semantic encoding with FACTORIZED EMBEDDINGS.
    
    Key properties:
    - Position-agnostic: Same token at different positions → same embedding
    - Factorized: Word embeddings are sums of morphological primitives
    - Zero OOV: Any valid inflection can be embedded compositionally
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.embedding = PaninianEmbedding(d_model)
        self.qkv_proj = QKVProjection(d_model, num_heads)
    
    def forward(self, batch: FactorizedTokenBatch) -> Phase2BOutput:
        """
        Args:
            batch: FactorizedTokenBatch with root_ids, type_ids, vibhakti_ids,
                   vacana_ids, purusa_ids — all shape (batch, seq)
        Returns:
            Phase2BOutput with embeddings and Q, K, V tensors
        """
        embeddings = self.embedding(
            batch["root_ids"],
            batch["type_ids"],
            batch["vibhakti_ids"],
            batch["vacana_ids"],
            batch["purusa_ids"],
        )
        qkv = self.qkv_proj(embeddings)
        return {"embeddings": embeddings, "qkv": qkv}
```

---

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| Unknown root | Stem not in root vocabulary | Use `<unk>` root embedding (ID=1) |
| Invalid grammatical ID | Tag value out of range | Clamp to valid range or use "none" ID |
| Shape mismatch | Inconsistent batch sizes | Validate all ID tensors have same shape |

---

## Test Specifications

### Unit Tests

```python
def test_no_positional_encoding():
    """Same token at different positions should have identical embedding."""
    engine = NeuralEngine(d_model=512, num_heads=8)
    
    # Same morphological analysis at positions 0 and 5
    batch = {
        "root_ids": torch.tensor([[42, 10, 20, 30, 40, 42]]),
        "type_ids": torch.tensor([[1, 0, 0, 0, 0, 1]]),
        "vibhakti_ids": torch.tensor([[0, 1, 2, 3, 4, 0]]),
        "vacana_ids": torch.tensor([[1, 1, 1, 1, 1, 1]]),
        "purusa_ids": torch.tensor([[1, 0, 0, 0, 0, 1]]),
    }
    output = engine(batch)
    
    # Position 0 and 5 have same factorized input → same embedding
    pos_0 = output["embeddings"][0, 0]
    pos_5 = output["embeddings"][0, 5]
    assert torch.allclose(pos_0, pos_5)

def test_factorized_embedding_composition():
    """Different inflections of same root should share root embedding component."""
    embedding = PaninianEmbedding(d_model=512)
    
    # "gacchati" (3rd person singular) vs "gacchāmi" (1st person singular)
    # Both share √gam (root_id=100)
    root_ids = torch.tensor([[100, 100]])
    type_ids = torch.tensor([[1, 1]])      # both tiṅanta
    vibhakti_ids = torch.tensor([[0, 0]])  # N/A for verbs
    vacana_ids = torch.tensor([[1, 1]])    # both singular
    purusa_ids = torch.tensor([[1, 3]])    # 3rd vs 1st person
    
    emb = embedding(root_ids, type_ids, vibhakti_ids, vacana_ids, purusa_ids)
    
    # The root component is identical; only puruṣa differs
    # Difference should equal (purusa_embed[1] - purusa_embed[3])
    diff = emb[0, 0] - emb[0, 1]
    expected_diff = embedding.purusa_embed.weight[1] - embedding.purusa_embed.weight[3]
    assert torch.allclose(diff, expected_diff)

def test_qkv_shapes():
    """Q, K, V should have correct shapes."""
    engine = NeuralEngine(d_model=512, num_heads=8)
    batch = {
        "root_ids": torch.tensor([[1, 2, 3, 4]]),
        "type_ids": torch.tensor([[0, 1, 0, 2]]),
        "vibhakti_ids": torch.tensor([[1, 0, 2, 0]]),
        "vacana_ids": torch.tensor([[1, 1, 1, 0]]),
        "purusa_ids": torch.tensor([[0, 1, 0, 0]]),
    }
    
    output = engine(batch)
    
    assert output["qkv"]["Q"].shape == (1, 8, 4, 64)  # (batch, heads, seq, head_dim)
    assert output["qkv"]["K"].shape == (1, 8, 4, 64)
    assert output["qkv"]["V"].shape == (1, 8, 4, 64)

def test_zero_oov_for_inflections():
    """
    Core property: The model can embed ANY valid Sanskrit inflection
    without an explicit vocabulary entry for that surface form.
    """
    embedding = PaninianEmbedding(d_model=512)
    
    # A hypothetical rare verb form the model never saw in training:
    # √kṛ (root 50) + laṅ (past) + 2nd person + dual
    rare_form = embedding(
        root_ids=torch.tensor([[50]]),
        type_ids=torch.tensor([[1]]),
        vibhakti_ids=torch.tensor([[0]]),
        vacana_ids=torch.tensor([[2]]),  # dual
        purusa_ids=torch.tensor([[2]]),  # 2nd person
    )
    
    # This should NOT raise any error and should produce a valid embedding
    assert rare_form.shape == (1, 1, 512)
    assert not torch.isnan(rare_form).any()

def test_batch_consistency():
    """Batched and individual processing should match."""
    engine = NeuralEngine(vocab_size=1000, d_model=512, num_heads=8)
    
    seq1 = torch.tensor([[1, 2, 3]])
    seq2 = torch.tensor([[4, 5, 6]])
    batch = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    out1 = engine(seq1)["embeddings"]
    out2 = engine(seq2)["embeddings"]
    out_batch = engine(batch)["embeddings"]
    
    assert torch.allclose(out_batch[0], out1[0])
    assert torch.allclose(out_batch[1], out2[0])
```

---

## Related Documents

- [Data Contracts](../types/data-contracts.md) — `QKVTensors`, `Phase2BOutput` definitions
- [Phase 1](phase1-morphology.md) — Input source
- [Phase 3](phase3-attention.md) — Consumes Q, K, V
- [Glossary](../GLOSSARY.md) — Position-agnostic embeddings definition

---

## Concrete Input/Output Examples

### Example 1: Token Encoding

**Input (MorphTokens from Phase 1):**
```python
tokens = [
    {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta", ...},
    {"surface": "gṛham", "stem": "gṛha", "type": "subanta", ...},
    {"surface": "gacchati", "stem": "gam", "type": "tinanta", ...}
]
```

**Tokenization Step:**
```python
# Tokenizer converts MorphTokens to integer IDs
token_ids = [2, 156, 892, 47, 3]  # [BOS, rāma, gṛha, gam, EOS]
type_ids  = [6, 0, 0, 1, 6]       # [unknown, subanta, subanta, tinanta, unknown]
```

**Output (Phase2BOutput):**
```python
{
    "embeddings": tensor([...]),  # Shape: (1, 5, 512)
    "qkv": {
        "Q": tensor([...]),       # Shape: (1, 8, 5, 64)
        "K": tensor([...]),       # Shape: (1, 8, 5, 64)
        "V": tensor([...])        # Shape: (1, 8, 5, 64)
    }
}
```

### Example 2: Training Data Format

In the training JSON, token_ids and type_ids are pre-computed:

```json
{
    "token_ids": [2, 1130, 17, 7, 234, 89, 3],
    "type_ids": [6, 0, 1, 2, 0, 0, 6],
    "seq_len": 7
}
```

**Interpretation:**
| Position | token_id | type_id | Meaning |
|----------|----------|---------|---------|
| 0 | 2 | 6 | `[BOS]` (unknown type) |
| 1 | 1130 | 0 | Noun (subanta) |
| 2 | 17 | 1 | Verb (tinanta) |
| 3 | 7 | 2 | Particle (avyaya) |
| 4 | 234 | 0 | Noun (subanta) |
| 5 | 89 | 0 | Noun (subanta) |
| 6 | 3 | 6 | `[EOS]` (unknown type) |

### Example 3: Embedding Computation

```python
# Given token_ids from training data
token_ids = torch.tensor([[2, 1130, 17, 7, 234, 89, 3]])  # (batch=1, seq=7)
type_ids = torch.tensor([[6, 0, 1, 2, 0, 0, 6]])          # (batch=1, seq=7)

# Token embedding: (batch, seq, d_model)
token_emb = embedding_layer(token_ids)  # (1, 7, 512)

# Type embedding: (batch, seq, d_model)
type_emb = type_embedding_layer(type_ids)  # (1, 7, 512)

# Combined embedding (sum)
embeddings = token_emb + type_emb  # (1, 7, 512)

# Q/K/V projection
Q = q_proj(embeddings).view(1, 7, 8, 64).transpose(1, 2)  # (1, 8, 7, 64)
K = k_proj(embeddings).view(1, 7, 8, 64).transpose(1, 2)  # (1, 8, 7, 64)
V = v_proj(embeddings).view(1, 7, 8, 64).transpose(1, 2)  # (1, 8, 7, 64)
```

### Example 4: Position-Agnostic Property

**Key Insight:** Same token at different positions → identical embedding

```python
# "रामः गच्छति रामः" (Rāmaḥ appears at positions 0 and 2)
token_ids = torch.tensor([[156, 47, 156]])  # rāma, gam, rāma

embeddings = model.phase2b(token_ids)["embeddings"]

# Position 0 and position 2 have IDENTICAL embeddings
assert torch.allclose(embeddings[0, 0], embeddings[0, 2])  # True!
```

This is fundamentally different from standard transformers where position encoding would make them different.

### Tensor Shape Summary

| Tensor | Shape | Config Values |
|--------|-------|---------------|
| `token_ids` | `(batch, seq)` | Variable |
| `type_ids` | `(batch, seq)` | Variable |
| `embeddings` | `(batch, seq, d_model)` | d_model=512 |
| `Q` | `(batch, heads, seq, head_dim)` | heads=8, head_dim=64 |
| `K` | `(batch, heads, seq, head_dim)` | heads=8, head_dim=64 |
| `V` | `(batch, heads, seq, head_dim)` | heads=8, head_dim=64 |
