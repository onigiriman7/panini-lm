"""
Vidyut-prakriya backend for morphological analysis.

vidyut-prakriya is a Rust library for Sanskrit morphology with Python bindings.
This module provides the wrapper for using it in Panini-LM.

Installation:
    pip install vidyut_py  # When available

See docs/integration/vidyut.md for integration details.
"""

from typing import List, Optional
import logging

from panini_lm.core.types import MorphToken, MorphAttributes
from panini_lm.core.exceptions import (
    SandhiResolutionError,
    UnknownTokenError,
    BackendUnavailableError,
)
from panini_lm.phase1_morphology.interface import MorphologicalAnalyzer


logger = logging.getLogger(__name__)


# Try to import vidyut
_VIDYUT_AVAILABLE = False
try:
    import vidyut_py
    _VIDYUT_AVAILABLE = True
except ImportError:
    vidyut_py = None  # type: ignore


class VidyutBackend(MorphologicalAnalyzer):
    """
    Morphological analyzer using vidyut-prakriya (Rust).
    
    This is the primary backend — faster and more accurate than
    the Python fallback.
    
    If vidyut_py is not installed, this backend reports as unavailable
    and the system falls back to HeritageBackend.
    """
    
    def __init__(self):
        if not _VIDYUT_AVAILABLE:
            logger.warning("vidyut_py not available, backend will report unavailable")
    
    @property
    def name(self) -> str:
        return "vidyut"
    
    @property
    def is_available(self) -> bool:
        return _VIDYUT_AVAILABLE
    
    def _ensure_available(self) -> None:
        """Raise error if backend not available."""
        if not self.is_available:
            raise BackendUnavailableError(
                backend="vidyut",
                reason="vidyut_py module not installed. Install with: pip install vidyut_py"
            )
    
    def sandhi_split(self, text: str) -> List[str]:
        """
        Split text at sandhi junctions using vidyut_py.
        
        Uses vidyut_py.Vyakarana for sandhi splitting.
        """
        self._ensure_available()
        
        try:
            # vidyut_py API (when available)
            # This is the expected API based on the library design
            results = vidyut_py.Vyakarana().segment(text)
            return [str(word) for word in results]
        except Exception as e:
            raise SandhiResolutionError(
                f"Sandhi resolution failed: {e}",
                input_text=text,
            ) from e
    
    def analyze_token(self, token: str) -> MorphToken:
        """
        Analyze a single token using vidyut_py.
        
        Extracts stem, type, and grammatical attributes.
        """
        self._ensure_available()
        
        try:
            # vidyut_py API for morphological analysis
            analysis = vidyut_py.Vyakarana().analyze(token)
            
            if not analysis:
                raise UnknownTokenError(f"No analysis found for token", token=token)
            
            # Take first analysis (most likely)
            best = analysis[0]
            
            # Map vidyut types to our types
            token_type = self._map_token_type(best.pos)
            attributes = self._extract_attributes(best)
            
            return {
                "surface": token,
                "stem": best.lemma or token,
                "type": token_type,
                "attributes": attributes,
            }
        except UnknownTokenError:
            raise
        except Exception as e:
            raise UnknownTokenError(
                f"Analysis failed for token: {e}",
                token=token
            ) from e
    
    def _map_token_type(self, pos: str) -> str:
        """Map vidyut POS tags to our token types."""
        mapping = {
            "noun": "subanta",
            "verb": "tinanta",
            "particle": "avyaya",
            "indeclinable": "avyaya",
            "participle": "krdanta",
            "adjective": "subanta",
            "pronoun": "subanta",
        }
        return mapping.get(pos.lower(), "unknown")
    
    def _extract_attributes(self, analysis) -> MorphAttributes:
        """Extract grammatical attributes from vidyut analysis."""
        attrs: MorphAttributes = {}
        
        # Map vidyut's attribute names to ours
        if hasattr(analysis, 'case') and analysis.case:
            attrs["vibhakti"] = analysis.case
        
        if hasattr(analysis, 'number') and analysis.number:
            number_map = {"singular": 1, "dual": 2, "plural": 3}
            attrs["vacana"] = number_map.get(analysis.number, 1)
        
        if hasattr(analysis, 'person') and analysis.person:
            person_map = {"third": 1, "second": 2, "first": 3}
            attrs["purusa"] = person_map.get(analysis.person, 1)
        
        if hasattr(analysis, 'gender') and analysis.gender:
            gender_map = {"masculine": "m", "feminine": "f", "neuter": "n"}
            attrs["linga"] = gender_map.get(analysis.gender, "m")
        
        if hasattr(analysis, 'tense') and analysis.tense:
            attrs["lakara"] = analysis.tense
        
        return attrs


# Singleton instance
_vidyut_instance: Optional[VidyutBackend] = None


def get_vidyut_backend() -> VidyutBackend:
    """Get the singleton VidyutBackend instance."""
    global _vidyut_instance
    if _vidyut_instance is None:
        _vidyut_instance = VidyutBackend()
    return _vidyut_instance
