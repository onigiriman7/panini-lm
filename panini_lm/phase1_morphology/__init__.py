"""
Phase 1: Morphological Ingestion

This module transforms raw Sanskrit text into structured morphological tokens.
It handles:
- Sandhi resolution (euphonic junction splitting)
- Samāsa decomposition (compound word splitting)
- Morphological analysis (grammatical attribute extraction)

See docs/phases/phase1-morphology.md for detailed documentation.
"""

from panini_lm.phase1_morphology.interface import MorphologicalAnalyzer
from panini_lm.phase1_morphology.orchestrator import (
    ingest_morphology,
    get_analyzer,
)

__all__ = [
    "MorphologicalAnalyzer",
    "ingest_morphology",
    "get_analyzer",
]
