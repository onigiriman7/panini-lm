"""
Configuration classes for Panini-LM.

Uses dataclasses for type-safe, immutable configuration with sensible defaults.
All hyperparameters are centralized here for easy tuning and documentation.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path


@dataclass(frozen=True)
class MorphologyConfig:
    """Configuration for Phase 1 Morphological Ingestion."""
    
    backend: Literal["vidyut", "heritage", "auto"] = "auto"
    """
    Morphological backend to use.
    - 'vidyut': Use vidyut-prakriya (Rust, faster)
    - 'heritage': Use sanskrit-heritage (Python, fallback)
    - 'auto': Try vidyut first, fall back to heritage
    """
    
    enable_samasa: bool = True
    """Whether to decompose compound words (samāsa)."""
    
    normalize_unicode: bool = True
    """Whether to NFC-normalize input before processing."""
    
    cache_analyses: bool = True
    """Whether to cache morphological analyses for repeated tokens."""
    
    max_sandhi_candidates: int = 5
    """Maximum sandhi resolution candidates to consider."""


@dataclass(frozen=True)
class SymbolicConfig:
    """Configuration for Phase 2A Symbolic Engine."""
    
    enable_subject_verb: bool = True
    """Enable kartā-kriyā (subject-verb) linking rules."""
    
    enable_object_verb: bool = True
    """Enable karma-kriyā (object-verb) linking rules."""
    
    enable_self_attention: bool = True
    """Always allow self-attention (token attends to itself)."""
    
    strict_agreement: bool = True
    """Require strict number/person agreement for links."""
    
    samsadhani_url: Optional[str] = None
    """URL for samsadhani Kāraka API (optional external service)."""
    
    samsadhani_timeout: float = 5.0
    """Timeout in seconds for samsadhani API calls."""


@dataclass(frozen=True)
class NeuralConfig:
    """Configuration for Phase 2B Neural Engine."""
    
    vocab_size: int = 50000
    """Size of the morphological vocabulary."""
    
    d_model: int = 512
    """Model dimension (embedding size)."""
    
    num_heads: int = 8
    """Number of attention heads."""
    
    use_positional_encoding: bool = False
    """
    Whether to add positional encoding.
    Default False: Sanskrit has free word order, position should not bias attention.
    """
    
    embedding_dropout: float = 0.1
    """Dropout rate for embeddings."""


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for Phase 3 Sparse Attention."""
    
    backend: Literal["triton", "pytorch", "auto"] = "auto"
    """
    Attention backend to use.
    - 'triton': Use Triton kernel (GPU, true FLOP savings)
    - 'pytorch': Use PyTorch (CPU/GPU, memory savings only)
    - 'auto': Try Triton first, fall back to PyTorch
    """
    
    block_size: int = 64
    """Block size for Triton block-sparse attention."""
    
    return_attention_weights: bool = False
    """Whether to return attention weights (memory intensive)."""
    
    attention_dropout: float = 0.1
    """Dropout rate for attention weights."""


@dataclass(frozen=True)
class FFNConfig:
    """Configuration for Phase 4 FFN."""
    
    expansion_factor: float = 1.5
    """
    FFN expansion factor. 
    Standard transformers use 4x, Panini uses 1.5-2x because:
    - Grammar is resolved in Phase 2A (saved capacity)
    - No positional encoding (no positional patterns to learn)
    """
    
    activation: Literal["swiglu", "gelu", "relu"] = "swiglu"
    """Activation function. SwiGLU recommended."""
    
    use_bias: bool = False
    """Whether to use bias in linear layers."""
    
    ffn_dropout: float = 0.1
    """Dropout rate after FFN."""


@dataclass(frozen=True)
class DecodingConfig:
    """Configuration for Phase 5 Constrained Decoding."""
    
    max_length: int = 128
    """Maximum generation length."""
    
    temperature: float = 1.0
    """Sampling temperature (1.0 = no scaling)."""
    
    top_k: Optional[int] = None
    """Top-k filtering (None = disabled)."""
    
    top_p: Optional[float] = None
    """Nucleus sampling threshold (None = disabled)."""
    
    enforce_grammar: bool = True
    """Whether to enforce grammatical constraints (main feature!)."""
    
    allow_fallback: bool = True
    """Whether to allow fallback when no tokens are grammatical."""


@dataclass(frozen=True)
class PaniniConfig:
    """
    Master configuration for the entire Panini-LM model.
    
    Example:
        config = PaniniConfig(
            neural=NeuralConfig(d_model=768, num_heads=12),
            ffn=FFNConfig(expansion_factor=2.0),
        )
    """
    
    # === Phase-specific configs ===
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    
    # === Global settings ===
    num_layers: int = 6
    """Number of transformer layers."""
    
    device: str = "cuda"
    """Device to run on ('cuda', 'cpu', 'mps')."""
    
    dtype: str = "float32"
    """Data type ('float32', 'float16', 'bfloat16')."""
    
    seed: int = 42
    """Random seed for reproducibility."""
    
    # === Paths ===
    vocab_path: Optional[str] = None
    """Path to vocabulary file."""
    
    checkpoint_path: Optional[str] = None
    """Path to model checkpoint."""
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.neural.d_model // self.neural.num_heads
    
    @property
    def d_ff(self) -> int:
        """Feed-forward intermediate dimension."""
        return int(self.neural.d_model * self.ffn.expansion_factor)
    
    def validate(self) -> None:
        """Validate configuration consistency."""
        assert self.neural.d_model % self.neural.num_heads == 0, \
            f"d_model ({self.neural.d_model}) must be divisible by num_heads ({self.neural.num_heads})"
        
        assert self.ffn.expansion_factor > 0, \
            f"expansion_factor must be positive, got {self.ffn.expansion_factor}"
        
        assert self.num_layers > 0, \
            f"num_layers must be positive, got {self.num_layers}"


# =============================================================================
# Preset Configurations
# =============================================================================

def get_small_config() -> PaniniConfig:
    """Small configuration for testing and debugging."""
    return PaniniConfig(
        neural=NeuralConfig(
            vocab_size=10000,
            d_model=256,
            num_heads=4,
        ),
        ffn=FFNConfig(expansion_factor=1.5),
        num_layers=4,
    )


def get_base_config() -> PaniniConfig:
    """Base configuration (default)."""
    return PaniniConfig()


def get_large_config() -> PaniniConfig:
    """Large configuration for production."""
    return PaniniConfig(
        neural=NeuralConfig(
            vocab_size=100000,
            d_model=1024,
            num_heads=16,
        ),
        ffn=FFNConfig(expansion_factor=2.0),
        num_layers=12,
    )
