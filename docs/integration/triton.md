# Triton Integration

> GPU-optimized block-sparse attention kernel.

---

## Overview

**Triton** is OpenAI's language for writing custom GPU kernels in Python. Panini-LM uses Triton to implement block-sparse attention that actually skips computation for invalid grammatical blocks.

- **Repository**: [github.com/openai/triton](https://github.com/openai/triton)
- **Purpose**: True FLOP reduction (not just memory saving)
- **Fallback**: PyTorch masked attention (saves memory, not FLOPs)

---

## Installation

```bash
pip install triton

# Verify
python -c "import triton; print(triton.__version__)"
```

### Requirements

- NVIDIA GPU with CUDA support
- CUDA 11.4+ recommended
- PyTorch 2.0+

---

## Why Triton?

### Standard PyTorch Attention

```python
# PyTorch with mask: still computes all N² scores, then masks
scores = Q @ K.T / sqrt(d)
scores = scores + mask  # mask has -inf for invalid
weights = softmax(scores)  # -inf → 0 probability
output = weights @ V
# FLOPs: O(N²) — same as without mask!
```

### Triton Block-Sparse

```python
# Triton: actually skips blocks where mask = -inf
@triton.jit
def sparse_attention_kernel(Q, K, V, M, Out, ...):
    block_i, block_j = tl.program_id(0), tl.program_id(1)
    
    # Check mask BEFORE loading data
    if M[block_i, block_j] == -inf:
        return  # No load, no compute!
    
    # Only compute for valid blocks
    q = tl.load(Q + block_i * stride)
    k = tl.load(K + block_j * stride)
    score = dot(q, k)
    # ...
# FLOPs: O(N·k) — massive reduction!
```

---

## Implementation

### Kernel Signature

```python
@triton.jit
def paninian_sparse_attention_kernel(
    Q_ptr, K_ptr, V_ptr,     # Input tensors
    M_ptr,                    # Adjacency matrix
    Out_ptr,                  # Output tensor
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 64,
):
    """
    Block-sparse attention using grammatical adjacency matrix M.
    
    Key optimization: Blocks where M[i,j] = -inf are completely skipped.
    No memory load, no computation — true FLOP savings.
    """
    # Get block indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_row = tl.program_id(2)
    
    # Pre-check: gather valid column indices for this row
    row_mask = tl.load(M_ptr + block_row * seq_len, 
                       mask=tl.arange(0, seq_len) < seq_len)
    valid_cols = tl.where(row_mask == 0.0)  # 0.0 = valid
    
    if tl.num_elements(valid_cols) == 0:
        # No valid connections — write zeros
        tl.store(Out_ptr + block_row * head_dim, tl.zeros((head_dim,)))
        return
    
    # Load Q for this position
    q_offset = batch_idx * num_heads * seq_len * head_dim + \
               head_idx * seq_len * head_dim + \
               block_row * head_dim
    q = tl.load(Q_ptr + q_offset + tl.arange(0, head_dim))
    
    # Compute attention only for valid columns
    max_score = -float('inf')
    scores = tl.zeros((seq_len,), dtype=tl.float32) - float('inf')
    
    for col in valid_cols:
        k_offset = batch_idx * num_heads * seq_len * head_dim + \
                   head_idx * seq_len * head_dim + \
                   col * head_dim
        k = tl.load(K_ptr + k_offset + tl.arange(0, head_dim))
        
        score = tl.sum(q * k) / tl.sqrt(float(head_dim))
        scores = tl.where(tl.arange(0, seq_len) == col, score, scores)
        max_score = tl.maximum(max_score, score)
    
    # Softmax (numerically stable)
    exp_scores = tl.exp(scores - max_score)
    sum_exp = tl.sum(exp_scores)
    weights = exp_scores / sum_exp
    
    # Weighted sum of V
    output = tl.zeros((head_dim,), dtype=tl.float32)
    for col in valid_cols:
        v_offset = batch_idx * num_heads * seq_len * head_dim + \
                   head_idx * seq_len * head_dim + \
                   col * head_dim
        v = tl.load(V_ptr + v_offset + tl.arange(0, head_dim))
        output += weights[col] * v
    
    # Store output
    out_offset = batch_idx * num_heads * seq_len * head_dim + \
                 head_idx * seq_len * head_dim + \
                 block_row * head_dim
    tl.store(Out_ptr + out_offset + tl.arange(0, head_dim), output)
```

### Python Wrapper

```python
import torch
import triton

def triton_sparse_attention(
    Q: torch.Tensor,
    K: torch.Tensor, 
    V: torch.Tensor,
    M: torch.Tensor
) -> torch.Tensor:
    """
    Sparse attention using Triton kernel.
    
    Args:
        Q: (batch, heads, seq, head_dim)
        K: (batch, heads, seq, head_dim)
        V: (batch, heads, seq, head_dim)
        M: (seq, seq) adjacency matrix
        
    Returns:
        Output: (batch, seq, d_model)
    """
    batch, heads, seq, head_dim = Q.shape
    
    # Allocate output
    output = torch.empty_like(Q)
    
    # Launch kernel
    grid = (batch, heads, seq)
    paninian_sparse_attention_kernel[grid](
        Q, K, V, M, output,
        seq_len=seq,
        head_dim=head_dim,
        num_heads=heads,
    )
    
    # Reshape: (batch, heads, seq, head_dim) → (batch, seq, d_model)
    output = output.transpose(1, 2).reshape(batch, seq, heads * head_dim)
    
    return output
```

---

## Performance

### FLOP Comparison

| Sequence Length | Dense FLOPs | Sparse FLOPs (k=3) | Reduction |
|-----------------|-------------|---------------------|-----------|
| 128 | 16,384 | 384 | 97.7% |
| 256 | 65,536 | 768 | 98.8% |
| 512 | 262,144 | 1,536 | 99.4% |
| 1024 | 1,048,576 | 3,072 | 99.7% |

### Memory Bandwidth

Triton kernel avoids loading K/V vectors for invalid blocks:

```
Standard: Load all K vectors → O(N² · d) memory reads
Sparse:   Load only valid K vectors → O(N · k · d) memory reads
```

---

## Fallback

When Triton is unavailable (CPU, no CUDA), use PyTorch:

```python
def pytorch_sparse_attention(Q, K, V, M):
    """Fallback using PyTorch masked attention."""
    scale = Q.shape[-1] ** -0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Apply mask (still computes full grid)
    scores = scores + M.unsqueeze(0).unsqueeze(0)
    
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    
    batch, heads, seq, head_dim = output.shape
    return output.transpose(1, 2).reshape(batch, seq, heads * head_dim)
```

---

## Testing

```python
def test_triton_correctness():
    """Verify Triton output matches PyTorch reference."""
    Q = torch.randn(2, 8, 64, 64, device='cuda')
    K = torch.randn(2, 8, 64, 64, device='cuda')
    V = torch.randn(2, 8, 64, 64, device='cuda')
    M = torch.zeros(64, 64, device='cuda')
    
    triton_out = triton_sparse_attention(Q, K, V, M)
    pytorch_out = pytorch_sparse_attention(Q, K, V, M)
    
    assert torch.allclose(triton_out, pytorch_out, atol=1e-4)
```

---

## Related Documentation

- [Phase 3 — Attention](../phases/phase3-attention.md)
- [Test Specifications](../testing/test-specifications.md)
