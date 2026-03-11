"""
Phase 2A: Symbolic Engine

This module generates the deterministic grammatical adjacency matrix M
from morphological tokens. The matrix encodes which token pairs can
attend to each other based on Pāṇinian grammar rules.

See docs/phases/phase2a-symbolic.md for detailed documentation.
"""

from panini_lm.phase2a_symbolic.matrix_builder import (
    build_adjacency_matrix,
    compute_adjacency_meta,
)
from panini_lm.phase2a_symbolic.rules import (
    GrammarRule,
    SubjectVerbRule,
    ObjectVerbRule,
    SelfAttentionRule,
    get_default_rules,
)

__all__ = [
    "build_adjacency_matrix",
    "compute_adjacency_meta",
    "GrammarRule",
    "SubjectVerbRule",
    "ObjectVerbRule",
    "SelfAttentionRule",
    "get_default_rules",
]
