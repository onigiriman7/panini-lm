"""
Phase 5 — Grammar-Constrained Decoding.

Ensures generated tokens are grammatically valid by:
1. Tracking morphological state (agreement requirements)
2. Computing valid next tokens based on grammar rules
3. Applying mask to logits before sampling

Key concept:
    During generation, not all vocabulary tokens are valid.
    For example, after generating a singular subject, only
    singular verbs should be allowed.

Components:
    - GrammarState: Tracks current grammatical constraints
    - GrammarMaskGenerator: Computes valid token masks
    - ConstrainedDecoder: Wrapper for constrained generation

Exports:
    - GrammarState: Morphological state tracker
    - compute_grammar_mask: Main mask computation
    - ConstrainedDecoder: Decoder with grammar constraints
"""

from panini_lm.phase5_decoding.state import GrammarState
from panini_lm.phase5_decoding.mask import compute_grammar_mask
from panini_lm.phase5_decoding.decoder import ConstrainedDecoder

__all__ = [
    "GrammarState",
    "compute_grammar_mask",
    "ConstrainedDecoder",
]
