"""
Phase 1 Orchestrator — Main entry point for morphological analysis.

Handles backend selection, fallback logic, and provides the main API.

Usage:
    from panini_lm.phase1_morphology import ingest_morphology
    
    result = ingest_morphology("rāmo'pi gacchati")
    print(result["tokens"])  # List of MorphToken
"""

from typing import Optional, Literal
import logging

from panini_lm.core.types import Phase1Output
from panini_lm.core.exceptions import BackendUnavailableError, MorphologyError
from panini_lm.core.config import MorphologyConfig
from panini_lm.phase1_morphology.interface import MorphologicalAnalyzer
from panini_lm.phase1_morphology.vidyut_backend import get_vidyut_backend
from panini_lm.phase1_morphology.heritage_backend import get_heritage_backend


logger = logging.getLogger(__name__)


# Cached analyzer instance
_cached_analyzer: Optional[MorphologicalAnalyzer] = None
_cached_backend: Optional[str] = None


def get_analyzer(
    backend: Literal["vidyut", "heritage", "auto"] = "auto"
) -> MorphologicalAnalyzer:
    """
    Get a morphological analyzer instance.
    
    Args:
        backend: Which backend to use
            - 'vidyut': Use vidyut-prakriya (Rust)
            - 'heritage': Use Python fallback
            - 'auto': Try vidyut first, fall back to heritage
            
    Returns:
        MorphologicalAnalyzer instance
        
    Raises:
        BackendUnavailableError: If requested backend not available
        
    Example:
        >>> analyzer = get_analyzer("auto")
        >>> analyzer.name
        'heritage'  # If vidyut not installed
    """
    global _cached_analyzer, _cached_backend
    
    # Return cached if same backend requested
    if _cached_analyzer is not None and _cached_backend == backend:
        return _cached_analyzer
    
    analyzer: Optional[MorphologicalAnalyzer] = None
    
    if backend == "vidyut":
        vidyut = get_vidyut_backend()
        if not vidyut.is_available:
            raise BackendUnavailableError(
                backend="vidyut",
                reason="vidyut_py module not installed"
            )
        analyzer = vidyut
        
    elif backend == "heritage":
        analyzer = get_heritage_backend()
        
    elif backend == "auto":
        # Try vidyut first
        vidyut = get_vidyut_backend()
        if vidyut.is_available:
            analyzer = vidyut
            logger.info(f"Using primary backend: vidyut")
        else:
            # Fall back to heritage
            analyzer = get_heritage_backend()
            logger.info(f"Vidyut unavailable, using fallback: heritage")
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Cache and return
    _cached_analyzer = analyzer
    _cached_backend = backend
    return analyzer


def ingest_morphology(
    text: str,
    config: Optional[MorphologyConfig] = None,
) -> Phase1Output:
    """
    Phase 1 main entry point: Transform raw Sanskrit text to morphological tokens.
    
    This is the main function for Phase 1. It:
    1. Selects the appropriate backend (vidyut or heritage)
    2. Normalizes the input text
    3. Resolves sandhi (euphonic junctions)
    4. Analyzes each token for morphological attributes
    
    Args:
        text: Raw Sanskrit text (Unicode UTF-8)
        config: Optional MorphologyConfig for customization
        
    Returns:
        Phase1Output containing:
        - tokens: List[MorphToken] — analyzed tokens
        - raw_input: str — original input
        - sandhi_splits: List[str] — intermediate sandhi splits
        
    Example:
        >>> result = ingest_morphology("rāmo'pi gacchati")
        >>> len(result["tokens"])
        3
        >>> result["tokens"][0]["stem"]
        'rāma'
        >>> result["tokens"][2]["type"]
        'tinanta'
        
    Raises:
        MorphologyError: For general analysis failures
        SandhiResolutionError: If sandhi cannot be resolved
    """
    if config is None:
        config = MorphologyConfig()
    
    # Get analyzer
    analyzer = get_analyzer(config.backend)
    
    logger.debug(f"Analyzing text with {analyzer.name}: {text[:50]}...")
    
    try:
        # Run full analysis pipeline
        result = analyzer.analyze(text)
        
        logger.debug(f"Analysis complete: {len(result['tokens'])} tokens")
        return result
        
    except (BackendUnavailableError, MorphologyError):
        raise
    except Exception as e:
        raise MorphologyError(f"Morphological analysis failed: {e}") from e


def clear_cache() -> None:
    """Clear the cached analyzer (useful for testing)."""
    global _cached_analyzer, _cached_backend
    _cached_analyzer = None
    _cached_backend = None
