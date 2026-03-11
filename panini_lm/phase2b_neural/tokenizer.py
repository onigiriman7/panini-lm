"""
Tokenizer for Pāṇinian LM.

Manages vocabulary from morphological stems and surfaces.
Maps MorphToken → integer IDs for embedding lookup.

Key design decisions:
- Uses STEMS as primary vocab (reduces vocabulary size)
- Surfaces used for OOV fallback
- Special tokens: [PAD], [UNK], [BOS], [EOS], [MASK]
"""

from __future__ import annotations

from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import json

from panini_lm.core.types import MorphToken


# Special token constants
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
MASK_TOKEN = "[MASK]"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN]


@dataclass
class PaniniTokenizer:
    """
    Vocabulary manager for Pāṇinian LM.
    
    Design:
    - Primary: Map stems to IDs (smaller vocab)
    - Fallback: Map surfaces for unseen stems
    - Type encoding: Separate embedding for token types
    
    Attributes:
        vocab: Mapping from token string to ID
        id_to_token: Reverse mapping
        type_vocab: Mapping from token type to ID
    """
    
    vocab: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)
    type_vocab: Dict[str, int] = field(default_factory=dict)
    
    # Special token IDs
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    mask_id: int = 4
    
    def __post_init__(self):
        """Initialize with special tokens if empty."""
        if not self.vocab:
            self._add_special_tokens()
    
    def _add_special_tokens(self) -> None:
        """Add special tokens to vocabulary."""
        for i, token in enumerate(SPECIAL_TOKENS):
            self.vocab[token] = i
            self.id_to_token[i] = token
        
        # Initialize type vocabulary
        self.type_vocab = {
            "subanta": 0,  # Noun
            "tinanta": 1,  # Verb
            "avyaya": 2,   # Indeclinable
            "krdanta": 3,  # Participle
            "taddhita": 4, # Derivative
            "samasa": 5,   # Compound
            "unknown": 6,  # Unknown type
        }
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    @property
    def num_types(self) -> int:
        """Return number of token types."""
        return len(self.type_vocab)
    
    def add_token(self, token: str) -> int:
        """
        Add a token to vocabulary if not present.
        
        Args:
            token: Token string (stem or surface)
            
        Returns:
            Token ID
        """
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.id_to_token[idx] = token
        return self.vocab[token]
    
    def encode_token(self, morph_token: MorphToken) -> int:
        """
        Encode a MorphToken to ID.
        
        Priority:
        1. Use stem if in vocab
        2. Use surface if stem not found
        3. Return UNK if neither found
        
        Args:
            morph_token: Morphological token
            
        Returns:
            Token ID
        """
        stem = morph_token.get("stem", "")
        surface = morph_token.get("surface", "")
        
        # Try stem first
        if stem and stem in self.vocab:
            return self.vocab[stem]
        
        # Try surface
        if surface and surface in self.vocab:
            return self.vocab[surface]
        
        return self.unk_id
    
    def encode_type(self, morph_token: MorphToken) -> int:
        """
        Encode token type to ID.
        
        Args:
            morph_token: Morphological token
            
        Returns:
            Type ID
        """
        token_type = morph_token.get("type", "unknown")
        return self.type_vocab.get(token_type, self.type_vocab["unknown"])
    
    def encode(
        self,
        tokens: List[MorphToken],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple[List[int], List[int]]:
        """
        Encode a sequence of MorphTokens.
        
        Args:
            tokens: List of morphological tokens
            add_bos: Whether to prepend BOS token
            add_eos: Whether to append EOS token
            
        Returns:
            Tuple of (token_ids, type_ids)
        """
        token_ids = []
        type_ids = []
        
        if add_bos:
            token_ids.append(self.bos_id)
            type_ids.append(self.type_vocab["unknown"])
        
        for token in tokens:
            token_ids.append(self.encode_token(token))
            type_ids.append(self.encode_type(token))
        
        if add_eos:
            token_ids.append(self.eos_id)
            type_ids.append(self.type_vocab["unknown"])
        
        return token_ids, type_ids
    
    def decode(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs back to strings.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of token strings
        """
        return [self.id_to_token.get(tid, UNK_TOKEN) for tid in token_ids]
    
    def build_vocab_from_tokens(
        self,
        tokens: Iterator[MorphToken],
        min_freq: int = 1,
    ) -> None:
        """
        Build vocabulary from token iterator.
        
        Args:
            tokens: Iterator of MorphTokens
            min_freq: Minimum frequency threshold
        """
        # Count frequencies
        stem_counts: Dict[str, int] = {}
        surface_counts: Dict[str, int] = {}
        
        for token in tokens:
            stem = token.get("stem", "")
            surface = token.get("surface", "")
            
            if stem:
                stem_counts[stem] = stem_counts.get(stem, 0) + 1
            if surface:
                surface_counts[surface] = surface_counts.get(surface, 0) + 1
        
        # Add stems first (priority)
        for stem, count in sorted(stem_counts.items(), key=lambda x: -x[1]):
            if count >= min_freq:
                self.add_token(stem)
        
        # Add surfaces that aren't already in vocab
        for surface, count in sorted(surface_counts.items(), key=lambda x: -x[1]):
            if count >= min_freq and surface not in self.vocab:
                self.add_token(surface)
    
    def save(self, path: Path) -> None:
        """Save tokenizer to file."""
        data = {
            "vocab": self.vocab,
            "type_vocab": self.type_vocab,
            "special_ids": {
                "pad": self.pad_id,
                "unk": self.unk_id,
                "bos": self.bos_id,
                "eos": self.eos_id,
                "mask": self.mask_id,
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "PaniniTokenizer":
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab=data["vocab"],
            type_vocab=data.get("type_vocab", {}),
        )
        tokenizer.id_to_token = {int(v): k for k, v in data["vocab"].items()}
        
        if "special_ids" in data:
            tokenizer.pad_id = data["special_ids"]["pad"]
            tokenizer.unk_id = data["special_ids"]["unk"]
            tokenizer.bos_id = data["special_ids"]["bos"]
            tokenizer.eos_id = data["special_ids"]["eos"]
            tokenizer.mask_id = data["special_ids"]["mask"]
        
        return tokenizer
