"""
Abstract interface for morphological analyzers.

All morphological backends (vidyut, heritage, etc.) must implement this interface.
This allows swapping backends without changing downstream code.
"""

from abc import ABC, abstractmethod
from typing import List

from panini_lm.core.types import MorphToken, Phase1Output


class MorphologicalAnalyzer(ABC):
    """
    Abstract base class for morphological analyzers.
    
    Implementations must provide:
    - sandhi_split(): Split text at euphonic junctions
    - analyze_token(): Extract morphological attributes from a single token
    - analyze(): Full pipeline from text to Phase1Output
    
    Example implementation:
        class MyAnalyzer(MorphologicalAnalyzer):
            def sandhi_split(self, text: str) -> List[str]:
                return my_sandhi_library.split(text)
            
            def analyze_token(self, token: str) -> MorphToken:
                result = my_morph_library.analyze(token)
                return {
                    "surface": token,
                    "stem": result.stem,
                    "type": result.pos,
                    "attributes": {...}
                }
    
    Usage:
        analyzer = get_analyzer()  # Auto-selects best backend
        result = analyzer.analyze("rāmo'pi gacchati")
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this analyzer backend (e.g., 'vidyut', 'heritage')."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend is available (dependencies installed)."""
        pass
    
    @abstractmethod
    def sandhi_split(self, text: str) -> List[str]:
        """
        Split text at sandhi (euphonic) junctions.
        
        Args:
            text: Raw Sanskrit text, possibly with sandhi
            
        Returns:
            List of individual words after sandhi resolution
            
        Example:
            >>> analyzer.sandhi_split("rāmo'pi")
            ["rāmaḥ", "api"]
            
        Raises:
            SandhiResolutionError: If sandhi cannot be resolved
        """
        pass
    
    @abstractmethod
    def analyze_token(self, token: str) -> MorphToken:
        """
        Analyze a single token (word) for morphological attributes.
        
        Args:
            token: A single Sanskrit word (post-sandhi resolution)
            
        Returns:
            MorphToken dict with surface, stem, type, and attributes
            
        Raises:
            UnknownTokenError: If token is not in vocabulary
            MorphologyError: For other analysis failures
        """
        pass
    
    def analyze(self, text: str) -> Phase1Output:
        """
        Full morphological analysis pipeline.
        
        This is the main entry point. Default implementation:
        1. Normalize Unicode (NFC)
        2. Split sandhi
        3. Analyze each token
        
        Args:
            text: Raw Sanskrit text
            
        Returns:
            Phase1Output with tokens, raw_input, and sandhi_splits
            
        Can be overridden for custom pipelines.
        """
        import unicodedata
        
        # Normalize Unicode
        normalized = unicodedata.normalize('NFC', text.strip())
        
        # Split sandhi
        splits = self.sandhi_split(normalized)
        
        # Analyze each token
        tokens = [self.analyze_token(s) for s in splits]
        
        return {
            "raw_input": text,
            "sandhi_splits": splits,
            "tokens": tokens,
        }
