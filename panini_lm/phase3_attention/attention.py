"""
Grammar-Constrained Sparse Attention.

Implements:
    Attn(Q, K, V, M) = softmax((Q @ K^T) / √d + M) @ V

Where M is the adjacency matrix from Phase 2A:
    M[i,j] = 0.0   → valid grammatical connection (normal attention)
    M[i,j] = -∞    → invalid connection (masked out after softmax)

This ensures attention only flows along grammatically valid paths.
"""

from __future__ import annotations

from typing import Optional, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from panini_lm.core.types import AdjacencyMatrix, QKVTensors
from panini_lm.core.config import AttentionConfig
from panini_lm.core.exceptions import ShapeMismatchError, KernelError

from panini_lm.phase3_attention.backend import (
    AttentionBackend,
    select_backend,
)


def sparse_attention_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor,
    dropout: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """
    PyTorch implementation of grammar-constrained attention.
    
    This uses standard dense attention computation but with the
    grammar mask applied. Memory savings from sparsity, but no
    FLOP savings (still computes all Q@K products).
    
    Args:
        Q: Query tensor, shape (batch, n_heads, seq_len, d_head)
        K: Key tensor, shape (batch, n_heads, seq_len, d_head)
        V: Value tensor, shape (batch, n_heads, seq_len, d_head)
        mask: Grammar adjacency matrix, shape (seq_len, seq_len)
              0.0 for valid, -inf for invalid
        dropout: Attention dropout probability
        training: Whether in training mode
    
    Returns:
        Attention output, shape (batch, n_heads, seq_len, d_head)
    """
    batch, n_heads, seq_len, d_head = Q.shape
    scale = 1.0 / math.sqrt(d_head)
    
    # Compute attention scores: Q @ K^T
    # Shape: (batch, n_heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Apply grammar mask
    # Broadcast mask to (1, 1, seq_len, seq_len) for batch and heads
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    scores = scores + mask
    
    # Softmax over keys dimension
    attn_weights = F.softmax(scores, dim=-1)
    
    # Dropout
    if dropout > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout, training=training)
    
    # Compute output: attn_weights @ V
    output = torch.matmul(attn_weights, V)
    
    return output


def sparse_attention_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor,
    block_size: int = 64,
    dropout: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """
    Triton block-sparse attention implementation.
    
    This achieves true FLOP savings by only computing attention
    for blocks that contain non-masked (valid) positions.
    
    NOTE: This is a stub that falls back to PyTorch.
    Full Triton kernel implementation requires significant
    development and testing.
    
    Args:
        Q, K, V: QKV tensors
        mask: Grammar mask
        block_size: Block size for sparse computation
        dropout: Attention dropout
        training: Whether in training mode
    
    Returns:
        Attention output
    """
    # TODO: Implement full Triton kernel
    # For now, fall back to PyTorch
    # This preserves the API for future optimization
    
    import logging
    logging.getLogger(__name__).debug(
        f"Triton sparse attention stub called with block_size={block_size}. "
        "Falling back to PyTorch implementation."
    )
    
    return sparse_attention_pytorch(Q, K, V, mask, dropout, training)


def sparse_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    adjacency: AdjacencyMatrix,
    config: Optional[AttentionConfig] = None,
    backend: Optional[Literal["pytorch", "triton", "auto"]] = None,
    training: bool = True,
) -> torch.Tensor:
    """
    Compute grammar-constrained sparse attention.
    
    This is the main entry point for Phase 3.
    
    Args:
        Q: Query tensor, shape (batch, n_heads, seq_len, d_head)
        K: Key tensor, shape (batch, n_heads, seq_len, d_head)
        V: Value tensor, shape (batch, n_heads, seq_len, d_head)
        adjacency: AdjacencyMatrix from Phase 2A
        config: Attention configuration
        backend: Backend preference ("pytorch", "triton", "auto")
        training: Whether in training mode
    
    Returns:
        Attention output, shape (batch, n_heads, seq_len, d_head)
    
    Raises:
        ShapeMismatchError: If tensor shapes don't match
    
    Example:
        >>> from panini_lm.phase3_attention import sparse_attention
        >>> 
        >>> # Q, K, V from Phase 2B
        >>> # adjacency from Phase 2A
        >>> output = sparse_attention(Q, K, V, adjacency)
    """
    config = config or AttentionConfig()
    backend_pref = backend or config.backend
    
    # Validate shapes
    _validate_shapes(Q, K, V, adjacency)
    
    # Get mask tensor
    mask = adjacency.matrix.to(Q.device)
    
    # Select backend
    selected = select_backend(backend_pref, Q.device)
    
    if selected == AttentionBackend.TRITON:
        return sparse_attention_triton(
            Q, K, V, mask,
            block_size=config.block_size,
            dropout=config.attention_dropout if training else 0.0,
            training=training,
        )
    else:
        return sparse_attention_pytorch(
            Q, K, V, mask,
            dropout=config.attention_dropout if training else 0.0,
            training=training,
        )


def _validate_shapes(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    adjacency: AdjacencyMatrix,
) -> None:
    """Validate that all tensor shapes are compatible."""
    if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
        raise ShapeMismatchError(
            f"Q, K, V must be 4D tensors. Got Q: {Q.dim()}, K: {K.dim()}, V: {V.dim()}"
        )
    
    if Q.shape != K.shape or K.shape != V.shape:
        raise ShapeMismatchError(
            f"Q, K, V must have same shape. Got Q: {Q.shape}, K: {K.shape}, V: {V.shape}"
        )
    
    batch, n_heads, seq_len, d_head = Q.shape
    mask_shape = adjacency.matrix.shape
    
    if mask_shape != (seq_len, seq_len):
        raise ShapeMismatchError(
            f"Adjacency matrix shape {mask_shape} doesn't match sequence length {seq_len}"
        )


class SparseAttentionLayer(nn.Module):
    """
    Grammar-constrained sparse attention as an nn.Module.
    
    Wraps the sparse_attention function for use in torch models.
    Can be used as a drop-in replacement for standard attention.
    
    Note: Does NOT include QKV projections — those are in Phase 2B.
    This module only handles the attention computation itself.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        backend: Literal["pytorch", "triton", "auto"] = "auto",
    ):
        """
        Initialize sparse attention layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Attention dropout probability
            backend: Backend preference
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.backend = backend
        
        # Config for attention computation
        self.config = AttentionConfig(
            backend=backend,
            attention_dropout=dropout,
        )
    
    @classmethod
    def from_config(cls, config: AttentionConfig, d_model: int, n_heads: int) -> "SparseAttentionLayer":
        """Create layer from config."""
        return cls(
            d_model=d_model,
            n_heads=n_heads,
            dropout=config.attention_dropout,
            backend=config.backend,
        )
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        adjacency: AdjacencyMatrix,
    ) -> torch.Tensor:
        """
        Compute sparse attention.
        
        Args:
            Q: Query tensor, shape (batch, n_heads, seq_len, d_head)
            K: Key tensor, shape (batch, n_heads, seq_len, d_head)
            V: Value tensor, shape (batch, n_heads, seq_len, d_head)
            adjacency: AdjacencyMatrix from Phase 2A
        
        Returns:
            Attention output, shape (batch, n_heads, seq_len, d_head)
        """
        return sparse_attention(
            Q, K, V, adjacency,
            config=self.config,
            training=self.training,
        )
    
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"dropout={self.dropout}, backend={self.backend}"
        )
