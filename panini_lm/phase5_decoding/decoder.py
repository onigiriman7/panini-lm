"""
Constrained Decoder.

Wraps a language model with grammar-constrained decoding.
Ensures generated text follows grammatical rules.

Supports:
- Greedy decoding with grammar constraints
- Beam search with grammar constraints
- Sampling with grammar bias
"""

from __future__ import annotations

from typing import Optional, List, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from panini_lm.core.types import MorphToken, GrammarMask
from panini_lm.phase5_decoding.state import GrammarState
from panini_lm.phase5_decoding.mask import (
    compute_grammar_mask,
    apply_grammar_mask,
    VocabMorphInfo,
)
from panini_lm.phase2b_neural import PaniniTokenizer


@dataclass
class GenerationOutput:
    """Output from constrained generation."""
    
    token_ids: List[int]
    """Generated token IDs."""
    
    tokens: List[MorphToken]
    """Generated MorphTokens (if available)."""
    
    final_state: GrammarState
    """Final grammar state."""
    
    is_complete: bool
    """Whether generation produced a complete sentence."""


class ConstrainedDecoder:
    """
    Grammar-constrained text decoder.
    
    Usage:
        decoder = ConstrainedDecoder(model, tokenizer)
        output = decoder.generate(prompt_ids, max_length=50)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PaniniTokenizer,
        vocab_info: Optional[VocabMorphInfo] = None,
    ):
        """
        Initialize constrained decoder.
        
        Args:
            model: Language model with forward() returning logits
            tokenizer: Tokenizer for vocabulary
            vocab_info: Pre-computed vocab morphological info
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_info = vocab_info or VocabMorphInfo(tokenizer)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        initial_state: Optional[GrammarState] = None,
        token_to_morph: Optional[Callable[[int], MorphToken]] = None,
    ) -> GenerationOutput:
        """
        Generate text with grammar constraints.
        
        Args:
            input_ids: Prompt token IDs, shape (1, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to sample (vs greedy)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            initial_state: Starting grammar state
            token_to_morph: Function to convert token ID to MorphToken
        
        Returns:
            GenerationOutput with generated sequence
        """
        device = input_ids.device
        state = initial_state or GrammarState.initial()
        
        generated_ids: List[int] = []
        generated_tokens: List[MorphToken] = []
        
        # Current sequence
        current_ids = input_ids.clone()
        
        for step in range(max_length):
            # Get model logits
            with torch.no_grad():
                outputs = self.model(current_ids)
                
            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Get logits for last position
            next_logits = logits[:, -1, :]  # (batch, vocab)
            
            # Apply grammar mask
            grammar_mask = compute_grammar_mask(
                state=state,
                tokenizer=self.tokenizer,
                vocab_info=self.vocab_info,
                eos_token_id=self.tokenizer.eos_id,
            )
            next_logits = apply_grammar_mask(next_logits, grammar_mask)
            
            # Temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Sample or argmax
            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                
                # Top-k filtering
                if top_k is not None:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(1, top_k_indices, top_k_probs)
                    probs = probs / probs.sum()
                
                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
                    probs = probs / probs.sum()
                
                next_id = torch.multinomial(probs, 1).item()
            else:
                next_id = next_logits.argmax(dim=-1).item()
            
            generated_ids.append(next_id)
            
            # Check for EOS
            if next_id == self.tokenizer.eos_id:
                break
            
            # Update state
            if token_to_morph is not None:
                morph_token = token_to_morph(next_id)
                generated_tokens.append(morph_token)
                state = state.update(morph_token)
            
            # Append to sequence
            next_tensor = torch.tensor([[next_id]], device=device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)
        
        return GenerationOutput(
            token_ids=generated_ids,
            tokens=generated_tokens,
            final_state=state,
            is_complete=state.is_complete_sentence(),
        )
    
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        initial_state: Optional[GrammarState] = None,
        token_to_morph: Optional[Callable[[int], MorphToken]] = None,
    ) -> GenerationOutput:
        """Convenience method for greedy decoding."""
        return self.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=False,
            initial_state=initial_state,
            token_to_morph=token_to_morph,
        )


def constrained_decode_step(
    logits: torch.Tensor,
    state: GrammarState,
    tokenizer: PaniniTokenizer,
    vocab_info: Optional[VocabMorphInfo] = None,
    temperature: float = 1.0,
) -> tuple[int, GrammarMask]:
    """
    Single decoding step with grammar constraints.
    
    Lower-level function for custom decoding loops.
    
    Args:
        logits: Model logits for next token, shape (vocab_size,)
        state: Current grammar state
        tokenizer: Tokenizer
        vocab_info: Vocabulary morphological info
        temperature: Sampling temperature
    
    Returns:
        Tuple of (selected_token_id, grammar_mask_used)
    """
    grammar_mask = compute_grammar_mask(
        state=state,
        tokenizer=tokenizer,
        vocab_info=vocab_info,
        eos_token_id=tokenizer.eos_id,
    )
    
    masked_logits = apply_grammar_mask(logits.unsqueeze(0), grammar_mask)
    
    if temperature != 1.0:
        masked_logits = masked_logits / temperature
    
    selected_id = masked_logits.argmax(dim=-1).item()
    
    return selected_id, grammar_mask
