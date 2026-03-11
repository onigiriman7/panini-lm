"""
Custom exception classes for Panini-LM.

Hierarchical exception structure allows catching at different granularities:
- Catch PaniniError for any Panini-LM error
- Catch phase-specific errors for targeted handling
- Catch specific errors for fine-grained control
"""

from typing import Optional, List, Any


class PaniniError(Exception):
    """
    Base exception for all Panini-LM errors.
    
    All custom exceptions inherit from this, allowing:
        try:
            ...
        except PaniniError as e:
            # Handle any Panini-LM error
    """
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# Phase 1: Morphological Ingestion Errors
# =============================================================================

class MorphologyError(PaniniError):
    """Base exception for Phase 1 morphological analysis errors."""
    pass


class SandhiResolutionError(MorphologyError):
    """
    Raised when sandhi (euphonic junction) cannot be resolved.
    
    This may occur when:
    - Input contains invalid character sequences
    - Ambiguous sandhi with no clear resolution
    - Unknown sandhi pattern not in the rules database
    
    Example:
        raise SandhiResolutionError(
            "Ambiguous sandhi at position 5",
            details={"input": "rāmo'pi", "position": 5, "candidates": ["rāmaḥ api", "rāmaḥ āpi"]}
        )
    """
    
    def __init__(
        self,
        message: str,
        input_text: Optional[str] = None,
        position: Optional[int] = None,
        candidates: Optional[List[str]] = None,
    ):
        details = {}
        if input_text:
            details["input"] = input_text
        if position is not None:
            details["position"] = position
        if candidates:
            details["candidates"] = candidates
        super().__init__(message, details)


class UnknownTokenError(MorphologyError):
    """
    Raised when a token is not found in the morphological database.
    
    Recovery options:
    - Use surface form as stem (graceful degradation)
    - Mark token type as "unknown"
    - Log warning and continue
    
    Example:
        raise UnknownTokenError("Token 'xyz' not in vocabulary", token="xyz")
    """
    
    def __init__(self, message: str, token: Optional[str] = None):
        details = {"token": token} if token else {}
        super().__init__(message, details)
        self.token = token


class BackendUnavailableError(MorphologyError):
    """
    Raised when the morphological backend (vidyut/heritage) is unavailable.
    
    This triggers the fallback mechanism:
    vidyut-prakriya → sanskrit-heritage → raise error
    """
    
    def __init__(self, backend: str, reason: Optional[str] = None):
        message = f"Backend '{backend}' is unavailable"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"backend": backend, "reason": reason})
        self.backend = backend


# =============================================================================
# Phase 2A: Symbolic Engine Errors
# =============================================================================

class SymbolicEngineError(PaniniError):
    """Base exception for Phase 2A symbolic engine errors."""
    pass


class RuleConflictError(SymbolicEngineError):
    """
    Raised when grammar rules produce conflicting results.
    
    Recovery: Apply rule priority ordering.
    """
    
    def __init__(
        self,
        message: str,
        rules: Optional[List[str]] = None,
        token_pair: Optional[tuple] = None,
    ):
        details = {}
        if rules:
            details["conflicting_rules"] = rules
        if token_pair:
            details["token_pair"] = token_pair
        super().__init__(message, details)


class InvalidGrammarError(SymbolicEngineError):
    """
    Raised when input violates fundamental grammatical constraints.
    
    Recovery: Return maximally sparse matrix, log warning.
    """
    
    def __init__(self, message: str, violations: Optional[List[str]] = None):
        details = {"violations": violations} if violations else {}
        super().__init__(message, details)


# =============================================================================
# Phase 2B: Neural Engine Errors
# =============================================================================

class NeuralEngineError(PaniniError):
    """Base exception for Phase 2B neural engine errors."""
    pass


class EmbeddingError(NeuralEngineError):
    """Raised when embedding computation fails."""
    
    def __init__(self, message: str, token_id: Optional[int] = None):
        details = {"token_id": token_id} if token_id is not None else {}
        super().__init__(message, details)


class ProjectionError(NeuralEngineError):
    """Raised when QKV projection fails."""
    pass


# =============================================================================
# Phase 3: Attention Errors
# =============================================================================

class AttentionError(PaniniError):
    """Base exception for Phase 3 attention errors."""
    pass


class KernelError(AttentionError):
    """
    Raised when Triton kernel compilation or execution fails.
    
    Recovery: Fall back to PyTorch implementation.
    """
    
    def __init__(self, message: str, kernel_name: Optional[str] = None):
        details = {"kernel": kernel_name} if kernel_name else {}
        super().__init__(message, details)


class ShapeMismatchError(AttentionError):
    """Raised when tensor shapes don't match expected dimensions."""
    
    def __init__(
        self,
        message: str,
        expected: Optional[tuple] = None,
        actual: Optional[tuple] = None,
    ):
        details = {}
        if expected:
            details["expected_shape"] = expected
        if actual:
            details["actual_shape"] = actual
        super().__init__(message, details)


# =============================================================================
# Phase 5: Decoding Errors
# =============================================================================

class DecodingError(PaniniError):
    """Base exception for Phase 5 constrained decoding errors."""
    pass


class EmptyGrammarMaskError(DecodingError):
    """
    Raised when no tokens are legal according to grammar constraints.
    
    Recovery: Relax constraints or force-allow top-k tokens.
    """
    
    def __init__(self, message: str, state: Optional[Any] = None):
        details = {"state": str(state)} if state else {}
        super().__init__(message, details)


class StateInconsistencyError(DecodingError):
    """Raised when morphological state becomes inconsistent."""
    
    def __init__(self, message: str, state: Optional[Any] = None):
        details = {"state": str(state)} if state else {}
        super().__init__(message, details)


# =============================================================================
# Utility Functions
# =============================================================================

def format_error_chain(error: Exception) -> str:
    """Format an exception with its full cause chain."""
    messages = [str(error)]
    cause = error.__cause__
    while cause:
        messages.append(f"  Caused by: {cause}")
        cause = cause.__cause__
    return "\n".join(messages)
