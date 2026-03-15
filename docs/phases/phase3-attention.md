# Phase 3 — Sparse Pāṇinian Attention

> O(N·k) attention using grammatical routing via Matrix M.

---

## Overview

Phase 3 combines the outputs of Phase 2A (Matrix M) and Phase 2B (Q, K, V) to compute **sparse attention** that only considers grammatically valid token relationships.

**Key innovation**: Instead of computing all N² attention scores, only compute scores for valid grammatical connections (typically k ≈ 2-3 per token), achieving **O(N·k)** complexity.

---

## Input/Output Contract

### Input

- **Q, K, V**: From Phase 2B, shape `(batch, heads, seq, head_dim)`
- **M**: Adjacency matrix from Phase 2A, shape `(seq, seq)`

### Output

- **Type**: `AttentionOutput` (see [data-contracts.md](../types/data-contracts.md))

```python
{
    "hidden_states": tensor(...),     # (batch, seq, d_model)
    "attention_weights": tensor(...)  # (batch, heads, seq, seq) [optional]
}
```

---

## Dependencies

- **Input**: Phase 2A (Matrix M), Phase 2B (Q, K, V)
- **External**: [Triton](../integration/triton.md) (GPU) or PyTorch (fallback)
- **Output consumers**: Phase 4 (Semantic Maturation)

---

## Implementation Details

### Attention Formula

Standard attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Pāṇinian sparse attention:
$$\text{Attention}(Q, K, V, M) = \text{softmax}\left(\text{Sparse}\left(\frac{QK^T}{\sqrt{d_k}}\right) + M\right) V$$

Where M[i,j] = -∞ causes position j's contribution to position i to become 0 after softmax.

### PyTorch Reference Implementation

```python
def pytorch_sparse_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    M: torch.Tensor
) -> torch.Tensor:
    """
    PyTorch fallback implementation.
    
    Note: Still computes full N² grid, then masks.
    Saves memory via masking but NOT compute FLOPs.
    
    Args:
        Q, K, V: (batch, heads, seq, head_dim)
        M: (seq, seq) adjacency matrix
        
    Returns:
        output: (batch, seq, d_model)
    """
    batch, heads, seq, head_dim = Q.shape
    d_model = heads * head_dim
    
    # Scaled dot-product attention
    scale = head_dim ** -0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)
    
    # Apply grammatical mask
    # M is (N, N), broadcast to (1, 1, N, N) for batch and heads
    scores = scores + M.unsqueeze(0).unsqueeze(0)
    
    # Softmax: -inf values become 0 probability
    weights = torch.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(weights, V)  # (B, H, N, D/H)
    
    # Reshape to (B, N, D)
    output = output.transpose(1, 2).reshape(batch, seq, d_model)
    
    return output
```

### Triton Optimized Kernel

See [Triton Integration](../integration/triton.md) for the full kernel implementation.

Key optimization: The Triton kernel checks M[i,j] **before** loading K[j], avoiding both memory access and computation for invalid blocks.

```python
# Pseudocode for Triton kernel
@triton.jit
def sparse_attention_kernel(Q, K, V, M, Out):
    i = tl.program_id(0)  # Query position
    
    # Find valid key positions for query i
    valid_j = tl.where(M[i, :] == 0.0)
    
    if len(valid_j) == 0:
        tl.store(Out[i], zeros)
        return
    
    q = tl.load(Q[i])
    scores = []
    
    # Only load and compute for valid positions
    for j in valid_j:
        k = tl.load(K[j])  # Only load valid K vectors
        scores.append(dot(q, k) / sqrt(d))
    
    weights = softmax(scores)
    
    output = zeros
    for idx, j in enumerate(valid_j):
        v = tl.load(V[j])
        output += weights[idx] * v
    
    tl.store(Out[i], output)
```

### Backend Selection

```python
def sparse_paninian_attention(
    qkv: QKVTensors,
    M: torch.Tensor,
    use_triton: bool = True
) -> AttentionOutput:
    """
    Dispatch to appropriate backend.
    """
    Q, K, V = qkv["Q"], qkv["K"], qkv["V"]
    
    # Try Triton first (true FLOP savings)
    if use_triton and torch.cuda.is_available():
        try:
            from panini_lm.kernels import triton_sparse_attention
            hidden = triton_sparse_attention(Q, K, V, M)
            return {"hidden_states": hidden, "attention_weights": None}
        except ImportError:
            pass
    
    # PyTorch fallback (memory savings only)
    hidden = pytorch_sparse_attention(Q, K, V, M)
    return {"hidden_states": hidden, "attention_weights": None}
```

---

## Performance Analysis

### Complexity Comparison

| Method | FLOPs | Memory | Notes |
|--------|-------|--------|-------|
| Dense attention | O(N² · d) | O(N²) | Standard transformer |
| PyTorch + mask | O(N² · d) | O(N) | Memory saved, not FLOPs |
| Triton sparse | O(N · k · d) | O(N · k) | True FLOP reduction |

### FLOP Reduction

For typical Sanskrit sentences with k ≈ 3 connections per token:

| N | Dense FLOPs | Sparse FLOPs | Reduction |
|---|-------------|--------------|-----------|
| 128 | 16,384 | 384 | 97.7% |
| 256 | 65,536 | 768 | 98.8% |
| 512 | 262,144 | 1,536 | 99.4% |

---

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| `KernelError` | Triton compilation failed | Fallback to PyTorch |
| Shape mismatch | Q/K/V shapes don't match M | Raise ValueError |
| All-inf mask row | Token has no valid connections | Use self-attention only |

---

## Test Specifications

### Correctness Tests

```python
def test_triton_matches_pytorch():
    """Triton output must match PyTorch reference."""
    Q = torch.randn(2, 8, 64, 64, device='cuda')
    K = torch.randn(2, 8, 64, 64, device='cuda')
    V = torch.randn(2, 8, 64, 64, device='cuda')
    M = torch.zeros(64, 64, device='cuda')
    
    triton_out = triton_sparse_attention(Q, K, V, M)
    pytorch_out = pytorch_sparse_attention(Q, K, V, M)
    
    assert torch.allclose(triton_out, pytorch_out, atol=1e-4)

def test_masked_positions_zero():
    """Masked positions should receive zero attention."""
    Q = torch.randn(1, 1, 4, 32)
    K = torch.randn(1, 1, 4, 32)
    V = torch.randn(1, 1, 4, 32)
    
    # Position 0 can only attend to position 0
    M = torch.full((4, 4), float('-inf'))
    M[0, 0] = 0.0
    
    _, weights = sparse_attention_with_weights(Q, K, V, M)
    
    assert weights[0, 0, 0, 0] == 1.0  # All weight on self
    assert weights[0, 0, 0, 1:].sum() < 1e-6  # Zero to others
```

### Performance Tests

```python
@pytest.mark.benchmark
def test_sparse_speedup(benchmark):
    """Sparse should be faster than dense for long sequences."""
    seq_len = 512
    Q = torch.randn(1, 8, seq_len, 64, device='cuda')
    K = torch.randn(1, 8, seq_len, 64, device='cuda')
    V = torch.randn(1, 8, seq_len, 64, device='cuda')
    
    # Very sparse mask (k=3)
    M = torch.full((seq_len, seq_len), float('-inf'), device='cuda')
    for i in range(seq_len):
        for j in range(max(0, i-1), min(seq_len, i+2)):
            M[i, j] = 0.0
    
    benchmark(triton_sparse_attention, Q, K, V, M)
```

---

## Related Documents

- [Data Contracts](../types/data-contracts.md) — `AttentionOutput` definition
- [Triton Integration](../integration/triton.md) — GPU kernel details
- [Phase 2A](phase2a-symbolic.md) — Source of Matrix M
- [Phase 2B](phase2b-neural.md) — Source of Q, K, V
- [Phase 4](phase4-ffn.md) — Consumes attention output

---

## Concrete Input/Output Examples

### Example 1: Combining Phase 2A and 2B Outputs

**From Phase 2A (Adjacency Matrix):**
```python
# "rāmaḥ gṛham gacchati" - Subject, Object, Verb
M = tensor([
    [0.0,  -inf, 0.0],   # rāmaḥ → self, verb
    [-inf, 0.0,  0.0],   # gṛham → self, verb
    [0.0,  0.0,  0.0]    # gacchati → all (verb sees all arguments)
])
```

**From Phase 2B (Q, K, V):**
```python
# Shape: (batch=1, heads=8, seq=3, head_dim=64)
Q = tensor([...])  # (1, 8, 3, 64)
K = tensor([...])  # (1, 8, 3, 64)
V = tensor([...])  # (1, 8, 3, 64)
```

**Attention Computation:**
```python
# Step 1: Compute raw attention scores
# scores = Q @ K^T / sqrt(d_k)
scores = torch.matmul(Q, K.transpose(-2, -1)) / 8.0  # (1, 8, 3, 3)

# Example raw scores (one head):
raw_scores = tensor([
    [1.2,  0.3,  0.8],   # rāmaḥ's scores to all
    [0.5,  1.1,  0.9],   # gṛham's scores to all
    [0.7,  0.6,  1.0]    # gacchati's scores to all
])

# Step 2: Add mask M
# scores + M = tensor([
#     [1.2,  -inf, 0.8],   # rāmaḥ: can see self, verb (not object)
#     [-inf, 1.1,  0.9],   # gṛham: can see self, verb (not subject)
#     [0.7,  0.6,  1.0]    # gacchati: can see all
# ])

masked_scores = scores + M.unsqueeze(0).unsqueeze(0)

# Step 3: Softmax (over last dim)
# -inf becomes 0 probability
weights = softmax(masked_scores, dim=-1)

# Result weights (one head):
# tensor([
#     [0.60, 0.00, 0.40],   # rāmaḥ: 60% self, 0% object, 40% verb
#     [0.00, 0.55, 0.45],   # gṛham: 0% subject, 55% self, 45% verb
#     [0.32, 0.28, 0.40]    # gacchati: distributed across all
# ])

# Step 4: Weighted sum of values
output = torch.matmul(weights, V)  # (1, 8, 3, 64)
```

**Output (AttentionOutput):**
```python
{
    "hidden_states": tensor([...]),  # Shape: (1, 3, 512)
    "attention_weights": None        # Optional, for debugging only
}
```

### Example 2: Effect of Masking

**Without mask (standard attention):**
```
rāmaḥ attends to: rāmaḥ(33%), gṛham(33%), gacchati(34%)
```

**With Pāṇinian mask:**
```
rāmaḥ attends to: rāmaḥ(60%), gṛham(0%), gacchati(40%)
                           ↑ masked out (no direct noun-noun relation)
```

The mask eliminates grammatically invalid attention paths.

### Example 3: Full Pipeline with Numeric Values

```python
import torch

# Config
batch, heads, seq, head_dim = 1, 2, 3, 4  # Small example
d_model = heads * head_dim  # 8

# Phase 2B outputs (toy values)
Q = torch.tensor([[
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],  # Head 0
    [[0, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0]]   # Head 1
]], dtype=torch.float32)  # (1, 2, 3, 4)

K = Q.clone()  # Same as Q for simplicity
V = torch.ones(1, 2, 3, 4)  # All ones

# Phase 2A output: Subject(0) → Verb(2), Object(1) → Verb(2)
M = torch.tensor([
    [0.0,  float('-inf'), 0.0],
    [float('-inf'), 0.0,  0.0],
    [0.0,  0.0,  0.0]
])

# Sparse attention
hidden = sparse_paninian_attention(Q, K, V, M)

# Output shape: (1, 3, 8)
assert hidden.shape == (1, 3, d_model)
```

### Example 4: Sparsity Impact on Compute

**Dense attention (no mask):**
```
For seq_len=100:
- Attention matrix: 100 × 100 = 10,000 score computations
- Each requires: Q[i] · K[j] = d_k multiplications
```

**Sparse Pāṇinian attention (k=3 average connections):**
```
For seq_len=100, k=3:
- Valid attention pairs: 100 × 3 = 300 score computations
- 97% FLOPs saved!
```

### Training Data → Attention Mask

The training data stores edges sparsely:

```json
{
    "adjacency_edges": [
        {"src": 0, "tgt": 0, "link_type": "sva-sambandha"},
        {"src": 0, "tgt": 2, "link_type": "kartā-kriyā"},
        {"src": 1, "tgt": 1, "link_type": "sva-sambandha"},
        {"src": 1, "tgt": 2, "link_type": "karma-kriyā"},
        {"src": 2, "tgt": 0, "link_type": "kartā-kriyā"},
        {"src": 2, "tgt": 1, "link_type": "karma-kriyā"},
        {"src": 2, "tgt": 2, "link_type": "sva-sambandha"}
    ],
    "seq_len": 3
}
```

**Conversion to dense matrix (in DataLoader):**
```python
def edges_to_matrix(edges: List[dict], seq_len: int) -> torch.Tensor:
    M = torch.full((seq_len, seq_len), float('-inf'))
    for edge in edges:
        M[edge["src"], edge["tgt"]] = 0.0
    return M
```
