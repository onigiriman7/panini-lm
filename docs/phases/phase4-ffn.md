# Phase 4 — Semantic Maturation (FFN)

> Compact SwiGLU feed-forward network with reduced expansion.

---

## Overview

Phase 4 applies a feed-forward network (FFN) to the attention output from Phase 3. The FFN "matures" the semantic representations by adding non-linearity and capacity for complex transformations.

**Key design choice**: Use **SwiGLU activation** with **1.5-2× expansion** (not the standard 4×) because:

1. Grammatical structure is resolved in Phases 2A/3 (saved capacity)
2. Position encodings removed (no positional patterns to learn)
3. Sanskrit compound semantics are structurally compositional

---

## Input/Output Contract

### Input

- **Type**: `AttentionOutput["hidden_states"]` from Phase 3
- **Shape**: `(batch, seq, d_model)`

### Output

- **Type**: `torch.Tensor`
- **Shape**: `(batch, seq, d_model)` — same as input

---

## Dependencies

- **Input**: Phase 3 output
- **External**: PyTorch (`torch.nn`)
- **Output consumers**: Phase 5 (or next layer's Phase 2B in stacked architecture)

---

## Implementation Details

### SwiGLU Activation

SwiGLU combines gating with the SiLU (Swish) activation:

$$\text{SwiGLU}(x) = \text{SiLU}(xW_1) \otimes (xW_2)$$

Where $\otimes$ is element-wise multiplication.

### Architecture

```
Input
  │
  ▼
┌─────────────────┐
│ RMSNorm         │  Pre-normalization (NOT LayerNorm)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│ W_gate│ │ W_up  │  Two parallel projections
│ →d_ff │ │ →d_ff │
└───┬───┘ └───┬───┘
    │         │
    ▼         │
┌───────┐     │
│ SiLU  │     │
└───┬───┘     │
    │         │
    ▼         ▼
┌─────────────────┐
│ Element-wise    │  Gated activation
│ Multiply        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ W_down          │  Project back to d_model
│ d_ff → d_model  │
└────────┬────────┘
         │
         ▼
Output
```

### Implementation

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network with reduced expansion.
    
    Standard transformer: d_ff = 4 * d_model
    Panini: d_ff = 1.5 ~ 2 * d_model (grammar resolves structure)
    """
    
    def __init__(
        self,
        d_model: int,
        expansion_factor: float = 1.5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Reduced expansion (not 4x!)
        d_ff = int(d_model * expansion_factor)
        
        self.norm = RMSNorm(d_model)
        
        # SwiGLU = SiLU(Wx) ⊙ Vx
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
        
        Returns:
            out: (batch, seq, d_model)
        """
        # Pre-norm
        normed = self.norm(x)
        
        # SwiGLU
        gate = F.silu(self.w_gate(normed))
        up = self.w_up(normed)
        hidden = gate * up
        
        # Project down
        out = self.w_down(hidden)
        out = self.dropout(out)
        
        # Residual connection
        return x + out
```

### Why 1.5-2× Expansion?

| Standard Transformer | Pāṇinian Transformer |
|---------------------|----------------------|
| 4× expansion needed for capacity | Grammatical capacity offloaded to M |
| Positional patterns encoded in FFN | No positions → less pattern diversity |
| Learns syntactic relationships | Syntax handled symbolically |

Empirical observation: 1.5-2× expansion maintains perplexity while reducing parameters by ~50%.

---

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| NaN values | Numerical instability | Check gradient clipping, reduce LR |
| OOM | Batch too large | Reduce batch size or use gradient checkpointing |

---

## Test Specifications

### Unit Tests

```python
def test_output_shape():
    """Output shape must match input shape."""
    ffn = SwiGLUFFN(d_model=512, expansion_factor=1.5)
    x = torch.randn(2, 64, 512)
    
    out = ffn(x)
    
    assert out.shape == x.shape

def test_residual_connection():
    """Zero initialization should pass through input."""
    ffn = SwiGLUFFN(d_model=512)
    
    # Zero out weights
    with torch.no_grad():
        ffn.w_down.weight.zero_()
    
    x = torch.randn(1, 10, 512)
    out = ffn(x)
    
    # With zeroed down projection, output should equal input
    assert torch.allclose(out, x)

def test_expansion_factor():
    """Intermediate dimension should respect expansion factor."""
    for factor in [1.5, 2.0, 4.0]:
        ffn = SwiGLUFFN(d_model=512, expansion_factor=factor)
        assert ffn.w_gate.out_features == int(512 * factor)

def test_rms_norm():
    """RMSNorm should normalize to unit RMS."""
    norm = RMSNorm(d_model=512)
    x = torch.randn(2, 10, 512) * 100  # Large values
    
    normed = norm(x)
    rms = torch.sqrt(torch.mean(normed ** 2, dim=-1))
    
    # Should be approximately 1.0
    assert torch.allclose(rms.mean(), torch.tensor(1.0), atol=0.1)
```

### Performance Comparison

```python
@pytest.mark.benchmark
def test_expansion_comparison(benchmark):
    """Compare 1.5x vs 4x expansion."""
    results = {}
    
    for factor in [1.5, 4.0]:
        ffn = SwiGLUFFN(d_model=512, expansion_factor=factor).cuda()
        x = torch.randn(32, 256, 512, device='cuda')
        
        time = benchmark(ffn, x)
        params = sum(p.numel() for p in ffn.parameters())
        
        results[f"{factor}x"] = {"time": time, "params": params}
    
    # 1.5x should have significantly fewer params
    assert results["1.5x"]["params"] < results["4.0x"]["params"] * 0.5
```

---

## Related Documents

- [Phase 3](phase3-attention.md) — Input source
- [Phase 5](phase5-decoding.md) — Next phase (or output)
- [Efficiency](../general/Efficiency.md) — Architecture efficiency analysis
- [Architecture](../general/Architecture.md) — Full pipeline context
