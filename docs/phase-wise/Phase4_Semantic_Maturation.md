# Phase 4 — Semantic Maturation

Purpose
- Process contextualized states produced by Phase 3 with compact FFNs dedicated to semantics.

Guidelines
- Use RMSNorm after attention outputs.
- Use a compact SwiGLU FFN with expansion factor in the range 1.5x–2x (not the standard 4x) to keep model small.

Pseudocode
```
def semantic_maturation(hidden):
    x = RMSNorm(hidden)
    x = SwiGLU_FFN(x)  # smaller expansion
    return x + hidden  # residual
```

Notes
- Because syntax is resolved externally, FFNs can focus on semantics and world knowledge; tune capacity accordingly.
