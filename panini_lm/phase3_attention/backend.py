"""
Backend selection for sparse attention computation.

Supports:
- PyTorch: Standard masked attention (memory savings, not FLOP savings)
- Triton: Block-sparse attention (true FLOP savings on GPU)

Auto mode tries Triton first, falls back to PyTorch.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Literal
import logging

import torch

logger = logging.getLogger(__name__)


class AttentionBackend(Enum):
    """Available attention computation backends."""
    
    PYTORCH = auto()
    """Standard PyTorch implementation. Works everywhere."""
    
    TRITON = auto()
    """Triton block-sparse kernel. GPU only, true FLOP savings."""


def is_triton_available() -> bool:
    """Check if Triton is available and usable."""
    try:
        import triton
        import triton.language as tl
        return torch.cuda.is_available()
    except ImportError:
        return False


def select_backend(
    preference: Literal["pytorch", "triton", "auto"] = "auto",
    device: torch.device | None = None,
) -> AttentionBackend:
    """
    Select attention backend based on preference and availability.
    
    Args:
        preference: User preference for backend
            - "pytorch": Always use PyTorch
            - "triton": Try Triton, fail if unavailable
            - "auto": Try Triton first, fallback to PyTorch
        device: Target device (used to check GPU availability)
    
    Returns:
        Selected AttentionBackend
    
    Raises:
        RuntimeError: If preference is "triton" but Triton unavailable
    """
    if preference == "pytorch":
        return AttentionBackend.PYTORCH
    
    triton_available = is_triton_available()
    
    if preference == "triton":
        if not triton_available:
            raise RuntimeError(
                "Triton backend requested but not available. "
                "Ensure Triton is installed and CUDA is available."
            )
        return AttentionBackend.TRITON
    
    # auto mode
    if triton_available:
        logger.info("Using Triton backend for sparse attention")
        return AttentionBackend.TRITON
    else:
        logger.info("Triton not available, using PyTorch backend")
        return AttentionBackend.PYTORCH


def get_block_size(
    seq_len: int,
    d_head: int,
    backend: AttentionBackend,
) -> int:
    """
    Determine optimal block size for sparse attention.
    
    For Triton, block size affects performance significantly.
    For PyTorch, this is informational only.
    
    Args:
        seq_len: Sequence length
        d_head: Head dimension
        backend: Selected backend
    
    Returns:
        Recommended block size
    """
    if backend == AttentionBackend.PYTORCH:
        return 1  # Not relevant for dense PyTorch
    
    # Triton block sizes (powers of 2)
    if d_head <= 64:
        block_k = 64
    else:
        block_k = 128
    
    # For short sequences, smaller blocks
    if seq_len < 128:
        return 32
    elif seq_len < 512:
        return 64
    else:
        return 128
