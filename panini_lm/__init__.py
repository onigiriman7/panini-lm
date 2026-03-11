"""
Panini-LM: Neuro-Symbolic Language Modeling via Pāṇinian Algebraic Priors.

A novel transformer architecture that decouples syntax from semantics using
Pāṇini's grammatical framework for Sanskrit language modeling.
"""

__version__ = "0.1.0"
__author__ = "Panini-LM Team"

from panini_lm.core.types import (
    MorphToken,
    MorphAttributes,
    Phase1Output,
    AdjacencyMatrix,
    QKVTensors,
    MorphologicalState,
    GrammarMask,
)
from panini_lm.core.config import PaniniConfig

__all__ = [
    "MorphToken",
    "MorphAttributes", 
    "Phase1Output",
    "AdjacencyMatrix",
    "QKVTensors",
    "MorphologicalState",
    "GrammarMask",
    "PaniniConfig",
]
