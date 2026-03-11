"""Tests for custom exceptions."""

import pytest

from panini_lm.core.exceptions import (
    PaniniError,
    SandhiResolutionError,
    UnknownTokenError,
    MorphologyError,
    RuleConflictError,
    InvalidGrammarError,
    BackendUnavailableError,
    KernelError,
    EmptyGrammarMaskError,
    format_error_chain,
)


class TestPaniniError:
    """Tests for base PaniniError."""
    
    def test_basic_error(self):
        """Create basic error with message."""
        error = PaniniError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.details == {}
    
    def test_error_with_details(self):
        """Create error with additional details."""
        error = PaniniError(
            "Processing failed",
            details={"phase": 1, "token": "xyz"}
        )
        assert "phase" in error.details
        assert error.details["token"] == "xyz"
        assert "Details:" in str(error)
    
    def test_error_hierarchy(self):
        """All errors should be catchable as PaniniError."""
        errors = [
            SandhiResolutionError("sandhi"),
            UnknownTokenError("unknown"),
            RuleConflictError("conflict"),
            InvalidGrammarError("invalid"),
            BackendUnavailableError("vidyut"),
            KernelError("triton"),
            EmptyGrammarMaskError("empty"),
        ]
        
        for error in errors:
            assert isinstance(error, PaniniError)


class TestSandhiResolutionError:
    """Tests for SandhiResolutionError."""
    
    def test_basic_sandhi_error(self):
        """Create basic sandhi error."""
        error = SandhiResolutionError("Ambiguous sandhi")
        assert isinstance(error, MorphologyError)
        assert isinstance(error, PaniniError)
    
    def test_sandhi_error_with_context(self):
        """Create sandhi error with full context."""
        error = SandhiResolutionError(
            "Ambiguous junction at position 5",
            input_text="rāmo'pi",
            position=5,
            candidates=["rāmaḥ api", "rāmaḥ āpi"]
        )
        assert error.details["input"] == "rāmo'pi"
        assert error.details["position"] == 5
        assert len(error.details["candidates"]) == 2


class TestUnknownTokenError:
    """Tests for UnknownTokenError."""
    
    def test_unknown_token(self):
        """Create unknown token error."""
        error = UnknownTokenError("Token not found", token="xyzabc")
        assert error.token == "xyzabc"
        assert error.details["token"] == "xyzabc"


class TestBackendUnavailableError:
    """Tests for BackendUnavailableError."""
    
    def test_backend_error(self):
        """Create backend unavailable error."""
        error = BackendUnavailableError("vidyut", reason="ImportError: no module")
        assert error.backend == "vidyut"
        assert "ImportError" in str(error)


class TestRuleConflictError:
    """Tests for RuleConflictError."""
    
    def test_rule_conflict(self):
        """Create rule conflict error."""
        error = RuleConflictError(
            "Multiple rules match",
            rules=["kartā-kriyā", "karma-kriyā"],
            token_pair=(0, 2)
        )
        assert "kartā-kriyā" in error.details["conflicting_rules"]


class TestKernelError:
    """Tests for KernelError."""
    
    def test_kernel_error(self):
        """Create kernel compilation error."""
        error = KernelError("Compilation failed", kernel_name="sparse_attention")
        assert error.details["kernel"] == "sparse_attention"


class TestErrorChain:
    """Tests for error chaining utilities."""
    
    def test_format_error_chain(self):
        """Format error with cause chain."""
        try:
            try:
                raise ValueError("Root cause")
            except ValueError as e:
                raise SandhiResolutionError("Higher level error") from e
        except SandhiResolutionError as e:
            formatted = format_error_chain(e)
            assert "Higher level error" in formatted
            assert "Root cause" in formatted
            assert "Caused by:" in formatted


class TestExceptionCatching:
    """Tests for exception hierarchy catching."""
    
    def test_catch_morphology_errors(self):
        """Catch all morphology errors with one handler."""
        errors_caught = 0
        
        for ErrorClass in [SandhiResolutionError, UnknownTokenError, BackendUnavailableError]:
            try:
                if ErrorClass == BackendUnavailableError:
                    raise ErrorClass("test")
                else:
                    raise ErrorClass("test message")
            except MorphologyError:
                errors_caught += 1
        
        assert errors_caught == 3
    
    def test_catch_all_panini_errors(self):
        """Catch all Panini-LM errors with base handler."""
        all_error_types = [
            SandhiResolutionError("a"),
            UnknownTokenError("b"),
            RuleConflictError("c"),
            InvalidGrammarError("d"),
            KernelError("e"),
            EmptyGrammarMaskError("f"),
        ]
        
        caught = 0
        for error in all_error_types:
            try:
                raise error
            except PaniniError:
                caught += 1
        
        assert caught == len(all_error_types)
