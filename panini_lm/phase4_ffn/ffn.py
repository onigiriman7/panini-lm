"""
Pāṇinian Feed-Forward Network.

Modified FFN with reduced expansion factor (1.5-2x instead of 4x).

Rationale:
    Standard transformers use 4x expansion because the FFN must learn
    both syntactic patterns AND semantic relationships. In Panini-LM,
    syntax is handled by grammar constraints (Phase 2A), so the FFN
    focuses purely on semantic transformation.

This allows:
    - Fewer parameters
    - Faster inference
    - Grammar rules guarantee correctness, FFN adds nuance
"""

from __future__ import annotations

from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from panini_lm.core.config import FFNConfig


class PaniniFeedForward(nn.Module):
    """
    Feed-forward network with reduced expansion factor.
    
    Architecture:
        hidden = GELU(x @ W1 + b1)
        hidden = Dropout(hidden)
        output = hidden @ W2 + b2
    
    Expansion factor defaults to 1.5x (vs standard 4x).
    """
    
    def __init__(
        self,
        d_model: int,
        expansion_factor: float = 1.5,
        dropout: float = 0.1,
        activation: Literal["gelu", "relu", "silu"] = "gelu",
        bias: bool = True,
    ):
        """
        Initialize FFN.
        
        Args:
            d_model: Model dimension
            expansion_factor: Hidden layer expansion (default 1.5x)
            dropout: Dropout probability
            activation: Activation function
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_hidden = int(d_model * expansion_factor)
        self.expansion_factor = expansion_factor
        
        # Linear layers
        self.W1 = nn.Linear(d_model, self.d_hidden, bias=bias)
        self.W2 = nn.Linear(self.d_hidden, d_model, bias=bias)
        
        # Activation
        self.activation = self._get_activation(activation)
        self.activation_name = activation
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.GELU())
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        if self.W1.bias is not None:
            nn.init.zeros_(self.W1.bias)
        if self.W2.bias is not None:
            nn.init.zeros_(self.W2.bias)
    
    @classmethod
    def from_config(cls, config: FFNConfig, d_model: int) -> "PaniniFeedForward":
        """Create FFN from config."""
        return cls(
            d_model=d_model,
            expansion_factor=config.expansion_factor,
            dropout=config.ffn_dropout,
            activation=config.activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor, shape (batch, seq_len, d_model)
        """
        # Up-projection + activation
        hidden = self.W1(x)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        
        # Down-projection
        output = self.W2(hidden)
        
        return output
    
    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"d_model={self.d_model}, d_hidden={self.d_hidden}, "
            f"expansion={self.expansion_factor:.1f}x, activation={self.activation_name}"
        )


class GatedFeedForward(nn.Module):
    """
    Gated FFN variant (similar to SwiGLU in LLaMA).
    
    Architecture:
        gate = Activation(x @ W_gate)
        up = x @ W_up
        hidden = gate * up
        output = hidden @ W_down
    
    Uses 2/3 of the hidden dimension for gate and up to match
    parameter count of standard FFN.
    """
    
    def __init__(
        self,
        d_model: int,
        expansion_factor: float = 1.5,
        dropout: float = 0.1,
        activation: Literal["gelu", "silu"] = "silu",
        bias: bool = False,
    ):
        """Initialize gated FFN."""
        super().__init__()
        
        self.d_model = d_model
        # Adjust hidden dim to account for gating
        self.d_hidden = int(d_model * expansion_factor * 2 / 3)
        self.expansion_factor = expansion_factor
        
        # Gate and up projections
        self.W_gate = nn.Linear(d_model, self.d_hidden, bias=bias)
        self.W_up = nn.Linear(d_model, self.d_hidden, bias=bias)
        self.W_down = nn.Linear(self.d_hidden, d_model, bias=bias)
        
        # Activation for gate
        self.activation = nn.SiLU() if activation == "silu" else nn.GELU()
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.W_gate, self.W_up, self.W_down]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gating.
        
        Args:
            x: Input, shape (batch, seq_len, d_model)
        
        Returns:
            Output, shape (batch, seq_len, d_model)
        """
        gate = self.activation(self.W_gate(x))
        up = self.W_up(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        output = self.W_down(hidden)
        
        return output
    
    def extra_repr(self) -> str:
        """String representation."""
        return f"d_model={self.d_model}, d_hidden={self.d_hidden}, expansion={self.expansion_factor:.1f}x (gated)"
