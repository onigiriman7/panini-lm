# Phase 2B — Neural Engine

> Position-agnostic semantic embedding and Q/K/V projection.

---

## Overview

Phase 2B is the **Neural Track** — it transforms morphological tokens into dense semantic representations without positional encoding.

**Critical design choice**: No positional encoding (RoPE, sinusoidal, etc.) because Sanskrit has free word order. Position should not bias attention.

---

## Input/Output Contract

### Input

- **Type**: `List[MorphToken]` from Phase 1
- **Used fields**: `stem`, `type`

### Output

- **Type**: `Phase2BOutput` (see [data-contracts.md](../types/data-contracts.md))
- **Contains**: Embeddings and Q, K, V tensors

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
| `embeddings` | (B, N, D) | Raw token embeddings |
| `Q`, `K`, `V` | (B, H, N, D/H) | Multi-head projections |

Where B=batch, N=seq_len, D=d_model, H=num_heads.

---

## Dependencies

- **Input**: Phase 1 output (`List[MorphToken]`)
- **External**: PyTorch (`torch.nn`)
- **Output consumers**: Phase 3 (Sparse Attention)

---

## Implementation Details

### Tokenization

Map (stem, type) pairs to integer IDs:

```python
class MorphTokenizer:
    """Tokenizer for morphological tokens."""
    
    def __init__(self, vocab_path: str):
        self.stem_to_id = load_vocab(vocab_path)
        self.type_to_offset = {
            "subanta": 0,
            "tinanta": 10000,
            "avyaya": 20000,
            "krdanta": 30000,
        }
        self.unk_id = 0
        self.pad_id = 1
    
    def encode(self, tokens: List[MorphToken]) -> torch.LongTensor:
        """Convert tokens to IDs."""
        ids = []
        for token in tokens:
            stem_id = self.stem_to_id.get(token["stem"], self.unk_id)
            type_offset = self.type_to_offset.get(token["type"], 0)
            ids.append(stem_id + type_offset)
        return torch.tensor(ids, dtype=torch.long)
```

### Embedding Layer

```python
class PositionAgnosticEmbedding(nn.Module):
    """
    Embedding layer WITHOUT positional encoding.
    
    Sanskrit has free word order — position should not bias attention.
    Word relationships are determined by morphology (Phase 2A), not position.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # NOTE: No self.pos_encoding!
    
    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # Just embedding lookup — no position added
        return self.embedding(token_ids)
```

### Q/K/V Projection

```python
class QKVProjection(nn.Module):
    """Project embeddings to Query, Key, Value tensors."""
    
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
            x: (batch, seq, d_model)
        Returns:
            Q, K, V each of shape (batch, heads, seq, head_dim)
        """
        batch, seq, _ = x.shape
        
        Q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        return {"Q": Q, "K": K, "V": V}
```

### Complete Phase 2B

```python
class NeuralEngine(nn.Module):
    """
    Phase 2B: Neural semantic encoding.
    
    Position-agnostic: Same token at different positions → same embedding.
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int):
        super().__init__()
        self.embedding = PositionAgnosticEmbedding(vocab_size, d_model)
        self.qkv_proj = QKVProjection(d_model, num_heads)
    
    def forward(self, token_ids: torch.LongTensor) -> Phase2BOutput:
        embeddings = self.embedding(token_ids)
        qkv = self.qkv_proj(embeddings)
        return {"embeddings": embeddings, "qkv": qkv}
```

---

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| Unknown token | Stem not in vocabulary | Use `<unk>` embedding |
| Shape mismatch | Inconsistent batch sizes | Validate input shapes |

---

## Test Specifications

### Unit Tests

```python
def test_no_positional_encoding():
    """Same token at different positions should have identical embedding."""
    engine = NeuralEngine(vocab_size=1000, d_model=512, num_heads=8)
    
    # Token 42 at positions 0 and 5
    ids = torch.tensor([[42, 10, 20, 30, 40, 42]])
    output = engine(ids)
    
    # Position 0 and 5 have same token → same embedding
    pos_0 = output["embeddings"][0, 0]
    pos_5 = output["embeddings"][0, 5]
    assert torch.allclose(pos_0, pos_5)

def test_qkv_shapes():
    """Q, K, V should have correct shapes."""
    engine = NeuralEngine(vocab_size=1000, d_model=512, num_heads=8)
    ids = torch.tensor([[1, 2, 3, 4]])  # batch=1, seq=4
    
    output = engine(ids)
    
    assert output["qkv"]["Q"].shape == (1, 8, 4, 64)  # (batch, heads, seq, head_dim)
    assert output["qkv"]["K"].shape == (1, 8, 4, 64)
    assert output["qkv"]["V"].shape == (1, 8, 4, 64)

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
