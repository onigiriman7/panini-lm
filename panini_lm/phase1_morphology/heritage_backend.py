"""
Sanskrit Heritage fallback backend for morphological analysis.

This is a pure Python implementation that serves as a fallback when
vidyut-prakriya is not available. It uses rule-based sandhi resolution
and a morphological database.

This backend is slower but provides reasonable accuracy for common cases.
See docs/integration/sanskrit-heritage.md for details.
"""

from typing import List, Dict, Optional, Tuple
import re
import logging

from panini_lm.core.types import MorphToken, MorphAttributes, TokenType
from panini_lm.core.exceptions import SandhiResolutionError, UnknownTokenError
from panini_lm.phase1_morphology.interface import MorphologicalAnalyzer


logger = logging.getLogger(__name__)


# =============================================================================
# Sandhi Rules Database
# =============================================================================

# Common sandhi patterns: (combined_form, split_result)
# These are the most frequent sandhi combinations in Sanskrit
SANDHI_RULES: List[Tuple[str, Tuple[str, str]]] = [
    # Visarga sandhi
    ("o'", ("aḥ", "a")),     # rāmo'pi → rāmaḥ + api
    ("o ", ("aḥ", "")),      # rāmo gacchati → rāmaḥ + gacchati
    ("āḥ", ("āḥ", "")),      # devāḥ → devāḥ (no change before consonants)
    
    # Vowel sandhi (savarna-dirgha)
    ("ā", ("a", "a")),       # a + a → ā
    ("ī", ("i", "i")),       # i + i → ī  
    ("ū", ("u", "u")),       # u + u → ū
    
    # Guna/Vrddhi sandhi
    ("e", ("a", "i")),       # a + i → e
    ("o", ("a", "u")),       # a + u → o
    ("ai", ("a", "e")),      # a + e → ai
    ("au", ("a", "o")),      # a + o → au
    
    # Consonant sandhi (common)
    ("ṃ", ("m", "")),        # Final m → ṃ before consonants
    ("cch", ("t", "ch")),    # t + ch → cch
    ("jj", ("t", "j")),      # t + j → jj
    
    # Avagraha (elision marker)
    ("'", ("", "a")),        # Elided a marker
]

# =============================================================================
# Morphological Database (Subset)
# =============================================================================

# Common vocabulary with morphological information
# Format: stem -> list of (surface, type, attributes)
MORPH_DATABASE: Dict[str, List[Tuple[str, TokenType, MorphAttributes]]] = {
    # Common nouns
    "rāma": [
        ("rāmaḥ", "subanta", {"vibhakti": 1, "vacana": 1, "linga": "m"}),
        ("rāmam", "subanta", {"vibhakti": 2, "vacana": 1, "linga": "m"}),
        ("rāmeṇa", "subanta", {"vibhakti": 3, "vacana": 1, "linga": "m"}),
        ("rāmāya", "subanta", {"vibhakti": 4, "vacana": 1, "linga": "m"}),
        ("rāmāt", "subanta", {"vibhakti": 5, "vacana": 1, "linga": "m"}),
        ("rāmasya", "subanta", {"vibhakti": 6, "vacana": 1, "linga": "m"}),
        ("rāme", "subanta", {"vibhakti": 7, "vacana": 1, "linga": "m"}),
    ],
    "deva": [
        ("devaḥ", "subanta", {"vibhakti": 1, "vacana": 1, "linga": "m"}),
        ("devam", "subanta", {"vibhakti": 2, "vacana": 1, "linga": "m"}),
        ("devāḥ", "subanta", {"vibhakti": 1, "vacana": 3, "linga": "m"}),
        ("devān", "subanta", {"vibhakti": 2, "vacana": 3, "linga": "m"}),
    ],
    "nara": [
        ("naraḥ", "subanta", {"vibhakti": 1, "vacana": 1, "linga": "m"}),
        ("naram", "subanta", {"vibhakti": 2, "vacana": 1, "linga": "m"}),
        ("narāḥ", "subanta", {"vibhakti": 1, "vacana": 3, "linga": "m"}),
    ],
    "gṛha": [
        ("gṛham", "subanta", {"vibhakti": 1, "vacana": 1, "linga": "n"}),
        ("gṛham", "subanta", {"vibhakti": 2, "vacana": 1, "linga": "n"}),
        ("gṛhe", "subanta", {"vibhakti": 7, "vacana": 1, "linga": "n"}),
    ],
    "phala": [
        ("phalam", "subanta", {"vibhakti": 1, "vacana": 1, "linga": "n"}),
        ("phalam", "subanta", {"vibhakti": 2, "vacana": 1, "linga": "n"}),
    ],
    
    # Common verbs (present tense, parasmaipada)
    "gam": [
        ("gacchati", "tinanta", {"purusa": 1, "vacana": 1, "lakara": "lat"}),
        ("gacchataḥ", "tinanta", {"purusa": 1, "vacana": 2, "lakara": "lat"}),
        ("gacchanti", "tinanta", {"purusa": 1, "vacana": 3, "lakara": "lat"}),
        ("gacchasi", "tinanta", {"purusa": 2, "vacana": 1, "lakara": "lat"}),
        ("gacchathaḥ", "tinanta", {"purusa": 2, "vacana": 2, "lakara": "lat"}),
        ("gacchatha", "tinanta", {"purusa": 2, "vacana": 3, "lakara": "lat"}),
        ("gacchāmi", "tinanta", {"purusa": 3, "vacana": 1, "lakara": "lat"}),
    ],
    "paś": [
        ("paśyati", "tinanta", {"purusa": 1, "vacana": 1, "lakara": "lat"}),
        ("paśyanti", "tinanta", {"purusa": 1, "vacana": 3, "lakara": "lat"}),
    ],
    "vad": [
        ("vadati", "tinanta", {"purusa": 1, "vacana": 1, "lakara": "lat"}),
        ("vadanti", "tinanta", {"purusa": 1, "vacana": 3, "lakara": "lat"}),
    ],
    "bhū": [
        ("bhavati", "tinanta", {"purusa": 1, "vacana": 1, "lakara": "lat"}),
        ("bhavanti", "tinanta", {"purusa": 1, "vacana": 3, "lakara": "lat"}),
    ],
    "as": [
        ("asti", "tinanta", {"purusa": 1, "vacana": 1, "lakara": "lat"}),
        ("staḥ", "tinanta", {"purusa": 1, "vacana": 2, "lakara": "lat"}),
        ("santi", "tinanta", {"purusa": 1, "vacana": 3, "lakara": "lat"}),
        ("asi", "tinanta", {"purusa": 2, "vacana": 1, "lakara": "lat"}),
        ("asmi", "tinanta", {"purusa": 3, "vacana": 1, "lakara": "lat"}),
    ],
    
    # Indeclinables (avyaya)
    "ca": [("ca", "avyaya", {})],
    "api": [("api", "avyaya", {})],
    "eva": [("eva", "avyaya", {})],
    "na": [("na", "avyaya", {})],
    "iti": [("iti", "avyaya", {})],
    "atra": [("atra", "avyaya", {})],
    "tatra": [("tatra", "avyaya", {})],
    "yatra": [("yatra", "avyaya", {})],
    "kutra": [("kutra", "avyaya", {})],
    "sarvatra": [("sarvatra", "avyaya", {})],
}

# Reverse lookup: surface form -> (stem, type, attributes)
_SURFACE_TO_ANALYSIS: Dict[str, Tuple[str, TokenType, MorphAttributes]] = {}

def _build_reverse_lookup():
    """Build reverse lookup table from surface forms to analysis."""
    for stem, forms in MORPH_DATABASE.items():
        for surface, token_type, attrs in forms:
            if surface not in _SURFACE_TO_ANALYSIS:
                _SURFACE_TO_ANALYSIS[surface] = (stem, token_type, attrs)

_build_reverse_lookup()


# =============================================================================
# Heritage Backend Implementation
# =============================================================================

class HeritageBackend(MorphologicalAnalyzer):
    """
    Pure Python morphological analyzer (fallback).
    
    Uses rule-based sandhi resolution and a morphological database.
    Less accurate than vidyut but always available.
    """
    
    def __init__(self):
        logger.info("Using HeritageBackend (Python fallback)")
    
    @property
    def name(self) -> str:
        return "heritage"
    
    @property
    def is_available(self) -> bool:
        return True  # Always available (pure Python)
    
    def sandhi_split(self, text: str) -> List[str]:
        """
        Split text at sandhi junctions using rule-based approach.
        
        Strategy:
        1. First try splitting on spaces
        2. For each word, check for sandhi patterns
        3. Apply sandhi rules to split combined forms
        """
        # Basic whitespace split first
        words = text.split()
        
        result = []
        for word in words:
            # Try to split further using sandhi rules
            splits = self._split_sandhi_word(word)
            result.extend(splits)
        
        return result
    
    def _split_sandhi_word(self, word: str) -> List[str]:
        """
        Try to split a single word that may contain sandhi.
        
        Uses pattern matching against known sandhi forms.
        """
        # Check for avagraha (apostrophe) - indicates sandhi
        if "'" in word:
            # Split at avagraha
            parts = word.split("'")
            result = []
            for i, part in enumerate(parts):
                if i == 0:
                    # First part: restore visarga if word ends in -o
                    if part.endswith("o"):
                        result.append(part[:-1] + "aḥ")
                    else:
                        result.append(part)
                else:
                    # Subsequent parts: restore initial vowel
                    if part and not part[0] in "aāiīuūeaioau":
                        part = "a" + part
                    result.append(part)
            return result
        
        # Check if word is in our database as-is
        if word in _SURFACE_TO_ANALYSIS or word.lower() in _SURFACE_TO_ANALYSIS:
            return [word]
        
        # Try common sandhi patterns
        for pattern, (first, second) in SANDHI_RULES:
            if pattern in word:
                idx = word.find(pattern)
                if idx > 0:  # Must have something before the pattern
                    before = word[:idx] + first
                    after = second + word[idx + len(pattern):]
                    if after:
                        return [before, after]
                    return [before]
        
        # No sandhi found, return as-is
        return [word]
    
    def analyze_token(self, token: str) -> MorphToken:
        """
        Analyze a single token using the morphological database.
        
        If token not found, returns with type='unknown'.
        """
        # Normalize (lowercase for lookup, preserve for surface)
        lookup_key = token
        
        # Try direct lookup
        if lookup_key in _SURFACE_TO_ANALYSIS:
            stem, token_type, attrs = _SURFACE_TO_ANALYSIS[lookup_key]
            return {
                "surface": token,
                "stem": stem,
                "type": token_type,
                "attributes": dict(attrs),  # Copy to avoid mutation
            }
        
        # Try lowercase
        if lookup_key.lower() in _SURFACE_TO_ANALYSIS:
            stem, token_type, attrs = _SURFACE_TO_ANALYSIS[lookup_key.lower()]
            return {
                "surface": token,
                "stem": stem,
                "type": token_type,
                "attributes": dict(attrs),
            }
        
        # Not found in database
        logger.debug(f"Token not in database: {token}")
        
        # Return with unknown type (graceful degradation)
        return {
            "surface": token,
            "stem": token,  # Use surface as stem
            "type": "unknown",
            "attributes": {},
        }


# Singleton instance
_heritage_instance: Optional[HeritageBackend] = None


def get_heritage_backend() -> HeritageBackend:
    """Get the singleton HeritageBackend instance."""
    global _heritage_instance
    if _heritage_instance is None:
        _heritage_instance = HeritageBackend()
    return _heritage_instance
