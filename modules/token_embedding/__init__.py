"""
Token embedding module for the Pāṇinian language model.

Converts Sanskrit words into 62-dimensional one-hot grammatical feature vectors.
"""

from modules.token_embedding.features import (
    PrimitiveType, Lakara, Purusha, Vacana, Prayoga, Pada,
    Vibhakti, Linga, Upasarga,
    FEATURE_ORDER, FEATURE_SIZES, FEATURE_OFFSETS, D_INPUT,
)
from modules.token_embedding.analyzer import GrammaticalVector, MorphAnalyzer
from modules.token_embedding.embedding import (
    encode_onehot, decode_onehot,
    assemble_sequence, assemble_batch,
    validate_onehot,
)

__all__ = [
    # Features
    "PrimitiveType", "Lakara", "Purusha", "Vacana", "Prayoga", "Pada",
    "Vibhakti", "Linga", "Upasarga",
    "FEATURE_ORDER", "FEATURE_SIZES", "FEATURE_OFFSETS", "D_INPUT",
    # Analyzer
    "GrammaticalVector", "MorphAnalyzer",
    # Embedding
    "encode_onehot", "decode_onehot",
    "assemble_sequence", "assemble_batch",
    "validate_onehot",
]
