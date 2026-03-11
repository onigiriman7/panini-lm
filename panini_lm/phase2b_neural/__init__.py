"""
Phase 2B — Neural Engine.

Processes morphological tokens through neural embedding and projection layers.
This is the **Neural Track** — containing learned parameters.

IMPORTANT: No positional encoding! Sanskrit has free word order,
so position should not bias attention.

Exports:
    - PaniniTokenizer: Vocabulary management
    - MorphAwareEmbedding: Embedding layer with morphological features
    - QKVProjection: Query/Key/Value projection module
    - process_neural: Main entry point
"""

from panini_lm.phase2b_neural.tokenizer import PaniniTokenizer
from panini_lm.phase2b_neural.embedding import MorphAwareEmbedding
from panini_lm.phase2b_neural.projection import QKVProjection
from panini_lm.phase2b_neural.processor import process_neural

__all__ = [
    "PaniniTokenizer",
    "MorphAwareEmbedding", 
    "QKVProjection",
    "process_neural",
]
