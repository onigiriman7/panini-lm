"""
Phase 4 — Semantic Maturation FFN Layer.

The feed-forward network that adds semantic nuance to grammatically-routed
representations from Phase 3.

Key design decision:
    - Expansion factor: 1.5-2x (not 4x)
    - Rationale: Syntax is handled by grammar constraints (Phase 2A),
      so FFN focuses on semantic transformation, not syntactic patterns

Architecture:
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

With residual connection and layer norm:
    output = LayerNorm(x + FFN(x))

Exports:
    - PaniniFeedForward: Main FFN module
    - TransformerBlock: Combined attention + FFN block
"""

from panini_lm.phase4_ffn.ffn import PaniniFeedForward
from panini_lm.phase4_ffn.block import TransformerBlock

__all__ = [
    "PaniniFeedForward",
    "TransformerBlock",
]
