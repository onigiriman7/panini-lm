"""
embedding.py — One-hot encoding and sequence assembly for grammatical vectors.

Converts GrammaticalVector instances into one-hot numpy arrays and assembles
them into context-window matrices suitable for the attention layer.

Key functions:
    encode_onehot()       — single vector → np.ndarray of shape (D_INPUT,)
    decode_onehot()       — np.ndarray → GrammaticalVector (inverse, for validation)
    assemble_sequence()   — list of vectors → (W, D_INPUT)
    assemble_batch()      — list of sequences → (B, W, D_INPUT)
"""

import numpy as np

from modules.token_embedding.features import (
    D_INPUT,
    FEATURE_SIZES,
    FEATURE_OFFSETS,
    FEATURE_ENUMS,
    FEATURE_ORDER,
)
from modules.token_embedding.analyzer import GrammaticalVector


# ===================================================================
# Encoding: GrammaticalVector → one-hot numpy array
# ===================================================================

def encode_onehot(gv: GrammaticalVector) -> np.ndarray:
    """Encode a GrammaticalVector as a one-hot vector of shape (D_INPUT,).

    Each of the 9 feature slots gets exactly one 1.
    NULL features have their 1 at index 0 of their slot.

    Args:
        gv: A GrammaticalVector with 9 integer feature indices.

    Returns:
        np.ndarray of shape (62,) with dtype float32.
    """
    vec = np.zeros(D_INPUT, dtype=np.float32)
    for feature_idx, offset, size in zip(gv.as_tuple(), FEATURE_OFFSETS, FEATURE_SIZES):
        assert 0 <= feature_idx < size, (
            f"Feature index {feature_idx} out of range [0, {size})"
        )
        vec[offset + feature_idx] = 1.0
    return vec


# ===================================================================
# Decoding: one-hot numpy array → GrammaticalVector (for validation)
# ===================================================================

def decode_onehot(vec: np.ndarray) -> GrammaticalVector:
    """Decode a one-hot vector back to a GrammaticalVector.

    Inverse of encode_onehot(). Useful for round-trip validation.

    Args:
        vec: np.ndarray of shape (D_INPUT,).

    Returns:
        A GrammaticalVector reconstructed from the one-hot encoding.
    """
    assert vec.shape == (D_INPUT,), f"Expected shape ({D_INPUT},), got {vec.shape}"

    indices = []
    for offset, size in zip(FEATURE_OFFSETS, FEATURE_SIZES):
        slot = vec[offset : offset + size]
        idx = int(np.argmax(slot))
        indices.append(idx)

    return GrammaticalVector(*indices)


# ===================================================================
# Sequence assembly
# ===================================================================

def assemble_sequence(vectors: list[GrammaticalVector], window_size: int) -> np.ndarray:
    """Stack a list of GrammaticalVectors into a context-window matrix.

    If len(vectors) < window_size, the sequence is left-padded with zeros.
    If len(vectors) > window_size, only the last window_size vectors are used.

    Args:
        vectors: List of GrammaticalVector instances.
        window_size: Context window size W.

    Returns:
        np.ndarray of shape (W, D_INPUT).
    """
    matrix = np.zeros((window_size, D_INPUT), dtype=np.float32)

    # Use only the last window_size vectors
    start = max(0, len(vectors) - window_size)
    selected = vectors[start:]

    # Right-align: pad on the left if fewer than window_size
    offset = window_size - len(selected)
    for i, gv in enumerate(selected):
        matrix[offset + i] = encode_onehot(gv)

    return matrix


def assemble_batch(
    sequences: list[list[GrammaticalVector]], window_size: int
) -> np.ndarray:
    """Assemble a batch of sequences into a 3D tensor.

    Args:
        sequences: List of B sequences, each a list of GrammaticalVectors.
        window_size: Context window size W.

    Returns:
        np.ndarray of shape (B, W, D_INPUT).
    """
    batch = np.stack([
        assemble_sequence(seq, window_size) for seq in sequences
    ])
    return batch


# ===================================================================
# Validation utilities
# ===================================================================

def validate_onehot(vec: np.ndarray) -> bool:
    """Check that a one-hot vector has exactly one 1 per feature slot.

    Returns True if valid, raises AssertionError otherwise.
    """
    assert vec.shape == (D_INPUT,), f"Shape mismatch: {vec.shape}"

    for i, (offset, size) in enumerate(zip(FEATURE_OFFSETS, FEATURE_SIZES)):
        slot = vec[offset : offset + size]
        ones = int(slot.sum())
        assert ones == 1, (
            f"Feature '{FEATURE_ORDER[i]}' has {ones} ones (expected 1) "
            f"at positions [{offset}:{offset + size}]"
        )

    return True
