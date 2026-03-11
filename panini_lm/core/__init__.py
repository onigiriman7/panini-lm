"""Core types, exceptions, and configuration for Panini-LM."""

from panini_lm.core.types import (
    MorphToken,
    MorphAttributes,
    Phase1Output,
    AdjacencyMatrix,
    AdjacencyMeta,
    QKVTensors,
    Phase2AOutput,
    Phase2BOutput,
    AttentionOutput,
    MorphologicalState,
    GrammarMask,
    GenerationOutput,
)
from panini_lm.core.exceptions import (
    PaniniError,
    SandhiResolutionError,
    UnknownTokenError,
    MorphologyError,
    RuleConflictError,
    InvalidGrammarError,
    BackendUnavailableError,
    KernelError,
)
from panini_lm.core.config import PaniniConfig

__all__ = [
    # Types
    "MorphToken",
    "MorphAttributes",
    "Phase1Output",
    "AdjacencyMatrix",
    "AdjacencyMeta",
    "QKVTensors",
    "Phase2AOutput",
    "Phase2BOutput",
    "AttentionOutput",
    "MorphologicalState",
    "GrammarMask",
    "GenerationOutput",
    # Exceptions
    "PaniniError",
    "SandhiResolutionError",
    "UnknownTokenError",
    "MorphologyError",
    "RuleConflictError",
    "InvalidGrammarError",
    "BackendUnavailableError",
    "KernelError",
    # Config
    "PaniniConfig",
]
