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
