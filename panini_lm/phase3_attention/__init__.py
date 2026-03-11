"""
Phase 3 — Sparse Grammar-Constrained Attention.

Combines Q, K, V from Phase 2B with adjacency matrix M from Phase 2A
to compute grammar-constrained attention.

Key insight:
    Attn = softmax((Q @ K^T) / √d + M)
    
Where M[i,j] = 0 for valid grammatical connections, -∞ for invalid.
This masks out attention to grammatically invalid positions.

Backends:
    - PyTorch: Standard implementation, works everywhere
    - Triton: Block-sparse GPU kernel for true FLOP savings

Exports:
    - sparse_attention: Main entry point
    - SparseAttentionLayer: nn.Module wrapper
    - select_backend: Backend selection utility
"""

from panini_lm.phase3_attention.attention import sparse_attention, SparseAttentionLayer
from panini_lm.phase3_attention.backend import select_backend, AttentionBackend

__all__ = [
    "sparse_attention",
    "SparseAttentionLayer",
    "select_backend",
    "AttentionBackend",
]
