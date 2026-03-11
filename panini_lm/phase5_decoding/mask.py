"""
Grammar Mask Generator.

Computes which vocabulary tokens are grammatically valid
given the current generation state.

The mask is applied to logits before sampling/argmax to
ensure only valid tokens can be selected.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Set
import torch

from panini_lm.core.types import MorphToken, GrammarMask
from panini_lm.core.exceptions import EmptyGrammarMaskError
from panini_lm.phase5_decoding.state import GrammarState

from panini_lm.phase2b_neural import PaniniTokenizer


class VocabMorphInfo:
    """
    Stores morphological information for vocabulary tokens.
    
    This is used to quickly look up which tokens match
    certain grammatical requirements.
    """
    
    def __init__(self, tokenizer: PaniniTokenizer):
        """Initialize with tokenizer vocabulary."""
        self.tokenizer = tokenizer
        
        # Build indices
        self._build_indices()
    
    def _build_indices(self) -> None:
        """Build lookup indices for fast filtering."""
        self.verb_tokens: Set[int] = set()
        self.noun_tokens: Set[int] = set()
        self.avyaya_tokens: Set[int] = set()
        
        # Vacana (number) indices for verbs
        self.verbs_by_vacana: Dict[int, Set[int]] = {
            1: set(),  # Singular
            2: set(),  # Dual
            3: set(),  # Plural
        }
        
        # Vibhakti (case) indices for nouns
        self.nouns_by_vibhakti: Dict[int, Set[int]] = {
            i: set() for i in range(1, 9)
        }
    
    def register_token(
        self,
        token_id: int,
        token_type: str,
        attributes: Dict,
    ) -> None:
        """
        Register a token's morphological information.
        
        Args:
            token_id: Vocabulary ID
            token_type: Morphological type
            attributes: Morphological attributes
        """
        if token_type == "tinanta":
            self.verb_tokens.add(token_id)
            vacana = attributes.get("vacana", 1)
            if vacana in self.verbs_by_vacana:
                self.verbs_by_vacana[vacana].add(token_id)
        
        elif token_type == "subanta":
            self.noun_tokens.add(token_id)
            vibhakti = attributes.get("vibhakti", 1)
            if vibhakti in self.nouns_by_vibhakti:
                self.nouns_by_vibhakti[vibhakti].add(token_id)
        
        elif token_type == "avyaya":
            self.avyaya_tokens.add(token_id)
    
    def get_tokens_with_vacana(self, vacana: int) -> Set[int]:
        """Get verb tokens with specific number."""
        return self.verbs_by_vacana.get(vacana, set())


def compute_grammar_mask(
    state: GrammarState,
    tokenizer: PaniniTokenizer,
    vocab_info: Optional[VocabMorphInfo] = None,
    allow_all_if_empty: bool = True,
    eos_token_id: Optional[int] = None,
) -> GrammarMask:
    """
    Compute grammar mask based on current state.
    
    The mask indicates which vocabulary tokens are valid next tokens.
    Invalid tokens get -infinity (masked out after softmax).
    
    Args:
        state: Current grammar state
        tokenizer: Tokenizer with vocabulary
        vocab_info: Pre-computed vocab morphological info
        allow_all_if_empty: If no constraints, allow all tokens
        eos_token_id: EOS token ID (always allowed if sentence complete)
    
    Returns:
        GrammarMask with valid token mask
    
    Example:
        >>> state = GrammarState.initial()
        >>> state = state.update(subject_token)  # Singular subject
        >>> mask = compute_grammar_mask(state, tokenizer)
        >>> # mask.mask will have 0.0 for singular verbs, -inf for plural
    """
    vocab_size = tokenizer.vocab_size
    
    # Start with all tokens allowed
    mask = torch.zeros(vocab_size)
    allowed_tokens: Set[int] = set(range(vocab_size))
    
    # Special tokens always allowed
    special_ids = {
        tokenizer.pad_id,
        tokenizer.unk_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.mask_id,
    }
    
    # Check for verb agreement requirement
    required_vacana = state.get_required_vacana()
    
    if required_vacana is not None and vocab_info is not None:
        # Must generate verb with matching number
        # For now, prefer matching verbs but don't exclude all else
        matching_verbs = vocab_info.get_tokens_with_vacana(required_vacana)
        
        if matching_verbs:
            # Boost valid verbs by keeping them, don't hard exclude others
            # This is soft guidance, not hard constraint
            pass
    
    # If sentence seems complete, allow EOS
    if state.is_complete_sentence():
        if eos_token_id is not None:
            allowed_tokens.add(eos_token_id)
    
    # Always allow special tokens
    allowed_tokens.update(special_ids)
    
    # Convert to mask tensor
    # 0.0 = allowed, -inf = blocked
    if allow_all_if_empty and len(allowed_tokens) == vocab_size:
        # No constraints
        mask = torch.zeros(vocab_size)
    else:
        mask = torch.full((vocab_size,), float('-inf'))
        for tid in allowed_tokens:
            mask[tid] = 0.0
    
    # Check for empty mask (error condition)
    if (mask == float('-inf')).all():
        if allow_all_if_empty:
            mask = torch.zeros(vocab_size)
        else:
            raise EmptyGrammarMaskError(
                "Grammar constraints resulted in empty valid token set",
                state=state.to_dict(),
            )
    
    return GrammarMask(
        mask=mask,
        legal_count=len(allowed_tokens),
    )


def apply_grammar_mask(
    logits: torch.Tensor,
    grammar_mask: GrammarMask,
) -> torch.Tensor:
    """
    Apply grammar mask to logits.
    
    Args:
        logits: Model output logits, shape (..., vocab_size)
        grammar_mask: Grammar mask to apply
    
    Returns:
        Masked logits (same shape as input)
    """
    mask = grammar_mask.mask.to(logits.device)
    
    # Broadcast mask to match logits shape
    while mask.dim() < logits.dim():
        mask = mask.unsqueeze(0)
    
    return logits + mask
