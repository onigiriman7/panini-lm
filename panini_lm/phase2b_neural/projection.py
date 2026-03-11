"""
Query, Key, Value Projection Module.

Projects embeddings to Q, K, V matrices for attention computation.
Supports multi-head attention with configurable number of heads.

These projections are fed into Phase 3 (Sparse Attention) along with
the adjacency matrix M from Phase 2A.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn

from panini_lm.core.types import QKVTensors
from panini_lm.core.config import NeuralConfig


class QKVProjection(nn.Module):
    """
    Linear projections for Query, Key, and Value matrices.
    
    Architecture:
    - Three separate linear layers: W_Q, W_K, W_V
    - Each projects from d_model to d_model
    - Output is split into n_heads for multi-head attention
    
    Design notes:
    - Fused QKV projection NOT used for clarity
    - Bias terms included by default
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize projections.
        
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to include bias terms
        """
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Separate projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.W_O = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize projection weights."""
        # Xavier initialization
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @classmethod
    def from_config(cls, config: NeuralConfig) -> "QKVProjection":
        """Create projections from config."""
        return cls(
            d_model=config.d_model,
            n_heads=config.num_heads,
            dropout=config.embedding_dropout,  # Use embedding dropout for now
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> QKVTensors | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q, K, V projections.
        
        Args:
            x: Input embeddings, shape (batch, seq_len, d_model)
            return_dict: If True, return QKVTensors dict
        
        Returns:
            If return_dict=True: QKVTensors dictionary
            Else: Tuple of (Q, K, V) tensors
            
            Each tensor has shape (batch, n_heads, seq_len, d_head)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        
        # Reshape for multi-head: (batch, seq, d_model) -> (batch, n_heads, seq, d_head)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        if return_dict:
            return {
                "Q": Q,
                "K": K,
                "V": V,
                "d_head": self.d_head,
                "n_heads": self.n_heads,
            }
        
        return Q, K, V
    
    def project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        """
        Project attention output back to d_model.
        
        Args:
            attn_output: Output from attention, shape (batch, n_heads, seq_len, d_head)
        
        Returns:
            Projected output, shape (batch, seq_len, d_model)
        """
        batch_size, n_heads, seq_len, d_head = attn_output.shape
        
        # Reshape: (batch, n_heads, seq, d_head) -> (batch, seq, d_model)
        output = attn_output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.W_O(output)
        output = self.dropout(output)
        
        return output


class QKVProjectionFused(nn.Module):
    """
    Fused QKV projection for efficiency.
    
    Instead of three separate linear layers, uses one large linear
    layer that produces Q, K, V in one operation. More efficient
    but less transparent.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """Initialize fused projection."""
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Fused projection: output 3 * d_model
        self.W_QKV = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.W_QKV.weight)
        nn.init.xavier_uniform_(self.W_O.weight)
        if self.W_QKV.bias is not None:
            nn.init.zeros_(self.W_QKV.bias)
        if self.W_O.bias is not None:
            nn.init.zeros_(self.W_O.bias)
    
    @classmethod
    def from_config(cls, config: NeuralConfig) -> "QKVProjectionFused":
        """Create from config."""
        return cls(
            d_model=config.d_model,
            n_heads=config.num_heads,
            dropout=config.embedding_dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> QKVTensors | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q, K, V with fused projection.
        
        Args:
            x: Input, shape (batch, seq_len, d_model)
            return_dict: Whether to return dict
        
        Returns:
            Q, K, V tensors, each shape (batch, n_heads, seq_len, d_head)
        """
        batch_size, seq_len, _ = x.shape
        
        # Fused projection
        qkv = self.W_QKV(x)  # (batch, seq, 3 * d_model)
        
        # Split into Q, K, V
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq, d_head)
        
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        if return_dict:
            return {
                "Q": Q,
                "K": K,
                "V": V,
                "d_head": self.d_head,
                "n_heads": self.n_heads,
            }
        
        return Q, K, V
    
    def project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Project attention output."""
        batch_size, n_heads, seq_len, d_head = attn_output.shape
        
        output = attn_output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_O(output)
        output = self.dropout(output)
        
        return output
