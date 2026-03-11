"""
Transformer Block combining Phase 3 (Attention) + Phase 4 (FFN).

This is a single transformer layer that:
1. Applies grammar-constrained sparse attention (Phase 3)
2. Applies semantic FFN (Phase 4)
3. Uses residual connections and layer normalization

Standard pre-norm architecture:
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
"""

from __future__ import annotations

from typing import Optional, Literal
import torch
import torch.nn as nn

from panini_lm.core.types import AdjacencyMatrix
from panini_lm.core.config import AttentionConfig, FFNConfig

from panini_lm.phase3_attention import SparseAttentionLayer
from panini_lm.phase4_ffn.ffn import PaniniFeedForward, GatedFeedForward
from panini_lm.phase2b_neural import QKVProjection


class TransformerBlock(nn.Module):
    """
    Single transformer block for Panini-LM.
    
    Architecture (pre-norm):
        1. x_norm = LayerNorm(x)
        2. Q, K, V = QKV_projection(x_norm)
        3. attn_out = SparseAttention(Q, K, V, adjacency)
        4. attn_out = Output_projection(attn_out)
        5. x = x + attn_out
        6. x_norm = LayerNorm(x)
        7. ffn_out = FFN(x_norm)
        8. x = x + ffn_out
    
    Note: This block includes QKV projections for convenience.
    In a full model, you might separate these concerns.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_factor: float = 1.5,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        attention_backend: Literal["pytorch", "triton", "auto"] = "auto",
        ffn_type: Literal["standard", "gated"] = "standard",
    ):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            expansion_factor: FFN expansion factor (default 1.5x)
            dropout: General dropout probability
            attention_dropout: Attention-specific dropout
            attention_backend: Attention computation backend
            ffn_type: Type of FFN ("standard" or "gated")
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # QKV projections
        self.qkv_projection = QKVProjection(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        
        # Sparse attention
        self.attention = SparseAttentionLayer(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout,
            backend=attention_backend,
        )
        
        # FFN
        if ffn_type == "gated":
            self.ffn = GatedFeedForward(
                d_model=d_model,
                expansion_factor=expansion_factor,
                dropout=dropout,
            )
        else:
            self.ffn = PaniniFeedForward(
                d_model=d_model,
                expansion_factor=expansion_factor,
                dropout=dropout,
            )
        
        self.dropout = nn.Dropout(dropout)
    
    @classmethod
    def from_configs(
        cls,
        attention_config: AttentionConfig,
        ffn_config: FFNConfig,
        d_model: int,
        n_heads: int,
    ) -> "TransformerBlock":
        """Create block from configs."""
        # Map activation to ffn_type
        use_gated = ffn_config.activation in ("swiglu", "silu")
        
        return cls(
            d_model=d_model,
            n_heads=n_heads,
            expansion_factor=ffn_config.expansion_factor,
            dropout=ffn_config.ffn_dropout,
            attention_dropout=attention_config.attention_dropout,
            attention_backend=attention_config.backend,
            ffn_type="gated" if use_gated else "standard",
        )
    
    def forward(
        self,
        x: torch.Tensor,
        adjacency: AdjacencyMatrix,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input embeddings, shape (batch, seq_len, d_model)
            adjacency: Grammar adjacency matrix from Phase 2A
        
        Returns:
            Output, shape (batch, seq_len, d_model)
        """
        # Pre-norm attention
        x_norm = self.norm1(x)
        Q, K, V = self.qkv_projection(x_norm)
        attn_out = self.attention(Q, K, V, adjacency)
        attn_out = self.qkv_projection.project_output(attn_out)
        x = x + self.dropout(attn_out)
        
        # Pre-norm FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x
    
    def extra_repr(self) -> str:
        """String representation."""
        return f"d_model={self.d_model}, n_heads={self.n_heads}"


class TransformerStack(nn.Module):
    """
    Stack of transformer blocks.
    
    This creates a full transformer encoder with multiple layers.
    """
    
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        expansion_factor: float = 1.5,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        attention_backend: Literal["pytorch", "triton", "auto"] = "auto",
        ffn_type: Literal["standard", "gated"] = "standard",
    ):
        """
        Initialize transformer stack.
        
        Args:
            n_layers: Number of transformer layers
            d_model: Model dimension
            n_heads: Number of attention heads
            expansion_factor: FFN expansion factor
            dropout: General dropout
            attention_dropout: Attention dropout
            attention_backend: Attention backend
            ffn_type: FFN type
        """
        super().__init__()
        
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                expansion_factor=expansion_factor,
                dropout=dropout,
                attention_dropout=attention_dropout,
                attention_backend=attention_backend,
                ffn_type=ffn_type,
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        adjacency: AdjacencyMatrix,
    ) -> torch.Tensor:
        """
        Forward through all layers.
        
        Args:
            x: Input embeddings, shape (batch, seq_len, d_model)
            adjacency: Grammar adjacency matrix
        
        Returns:
            Output, shape (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, adjacency)
        
        x = self.final_norm(x)
        
        return x
