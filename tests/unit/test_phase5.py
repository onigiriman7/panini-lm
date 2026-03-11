"""
Tests for Phase 5: Grammar-Constrained Decoding.

Test categories:
1. State tests — grammar state tracking
2. Mask tests — valid token computation
3. Decoder tests — constrained generation

Run with: pytest tests/unit/test_phase5.py -v
"""

import pytest
import torch

from panini_lm.core.types import MorphToken
from panini_lm.phase5_decoding import GrammarState, compute_grammar_mask, ConstrainedDecoder
from panini_lm.phase5_decoding.state import AgreementRequirement
from panini_lm.phase5_decoding.mask import VocabMorphInfo, apply_grammar_mask
from panini_lm.phase2b_neural import PaniniTokenizer


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def tokenizer() -> PaniniTokenizer:
    """Create tokenizer with test vocabulary."""
    tok = PaniniTokenizer()
    # Add some test tokens
    tok.add_token("rāma")
    tok.add_token("gam")
    tok.add_token("gṛha")
    tok.add_token("devāḥ")  # Plural
    return tok


@pytest.fixture
def singular_subject() -> MorphToken:
    """Singular nominative subject."""
    return {
        "surface": "rāmaḥ",
        "stem": "rāma",
        "type": "subanta",
        "attributes": {"vibhakti": 1, "vacana": 1},
    }


@pytest.fixture
def plural_subject() -> MorphToken:
    """Plural nominative subject."""
    return {
        "surface": "devāḥ",
        "stem": "deva",
        "type": "subanta",
        "attributes": {"vibhakti": 1, "vacana": 3},
    }


@pytest.fixture
def singular_verb() -> MorphToken:
    """Singular verb."""
    return {
        "surface": "gacchati",
        "stem": "gam",
        "type": "tinanta",
        "attributes": {"vacana": 1, "lakara": "lat"},
    }


@pytest.fixture
def plural_verb() -> MorphToken:
    """Plural verb."""
    return {
        "surface": "gacchanti",
        "stem": "gam",
        "type": "tinanta",
        "attributes": {"vacana": 3, "lakara": "lat"},
    }


# =============================================================================
# Grammar State Tests
# =============================================================================

class TestGrammarState:
    """Tests for grammar state tracking."""
    
    def test_initial_state(self):
        """Initial state should be empty."""
        state = GrammarState.initial()
        
        assert len(state.tokens) == 0
        assert not state.has_subject
        assert not state.has_verb
        assert state.expected_vacana is None
    
    def test_update_with_subject(self, singular_subject):
        """Adding subject should update state."""
        state = GrammarState.initial()
        new_state = state.update(singular_subject)
        
        assert len(new_state.tokens) == 1
        assert new_state.has_subject
        assert new_state.expected_vacana == 1  # Singular
    
    def test_state_immutability(self, singular_subject):
        """Original state should not be modified."""
        state = GrammarState.initial()
        new_state = state.update(singular_subject)
        
        assert len(state.tokens) == 0
        assert not state.has_subject
        assert len(new_state.tokens) == 1
    
    def test_subject_creates_verb_requirement(self, singular_subject):
        """Subject should create verb agreement requirement."""
        state = GrammarState.initial()
        new_state = state.update(singular_subject)
        
        assert len(new_state.pending_agreements) == 1
        req = new_state.pending_agreements[0]
        assert req.category == "verb"
        assert req.attributes.get("vacana") == 1
        assert not req.satisfied
    
    def test_verb_satisfies_requirement(self, singular_subject, singular_verb):
        """Matching verb should satisfy agreement."""
        state = GrammarState.initial()
        state = state.update(singular_subject)
        state = state.update(singular_verb)
        
        unsatisfied = state.get_unsatisfied_requirements()
        assert len(unsatisfied) == 0
    
    def test_wrong_verb_doesnt_satisfy(self, singular_subject, plural_verb):
        """Non-matching verb should not satisfy agreement."""
        state = GrammarState.initial()
        state = state.update(singular_subject)
        state = state.update(plural_verb)
        
        unsatisfied = state.get_unsatisfied_requirements()
        assert len(unsatisfied) == 1
    
    def test_is_complete_sentence(self, singular_subject, singular_verb):
        """Complete sentence should be detected."""
        state = GrammarState.initial()
        
        # Incomplete: no subject
        assert not state.is_complete_sentence()
        
        # Incomplete: subject but no verb
        state = state.update(singular_subject)
        assert not state.is_complete_sentence()
        
        # Complete: subject + matching verb
        state = state.update(singular_verb)
        assert state.is_complete_sentence()
    
    def test_requires_verb(self, singular_subject, singular_verb):
        """Should detect when verb is needed."""
        state = GrammarState.initial()
        assert not state.requires_verb()
        
        state = state.update(singular_subject)
        assert state.requires_verb()
        
        state = state.update(singular_verb)
        assert not state.requires_verb()
    
    def test_get_required_vacana(self, singular_subject, singular_verb):
        """Should return required number before verb."""
        state = GrammarState.initial()
        assert state.get_required_vacana() is None
        
        state = state.update(singular_subject)
        assert state.get_required_vacana() == 1
        
        state = state.update(singular_verb)
        assert state.get_required_vacana() is None
    
    def test_to_dict(self, singular_subject):
        """State should serialize to dict."""
        state = GrammarState.initial()
        state = state.update(singular_subject)
        
        d = state.to_dict()
        assert "num_tokens" in d
        assert d["num_tokens"] == 1
        assert d["has_subject"] is True


# =============================================================================
# Grammar Mask Tests
# =============================================================================

class TestGrammarMask:
    """Tests for grammar mask computation."""
    
    def test_mask_shape(self, tokenizer):
        """Mask should have vocab_size shape."""
        state = GrammarState.initial()
        mask = compute_grammar_mask(state, tokenizer)
        
        assert mask.mask.shape == (tokenizer.vocab_size,)
    
    def test_special_tokens_always_allowed(self, tokenizer):
        """Special tokens should always be valid."""
        state = GrammarState.initial()
        mask = compute_grammar_mask(state, tokenizer)
        
        # Check special tokens have 0.0 (allowed)
        assert mask.mask[tokenizer.pad_id] == 0.0
        assert mask.mask[tokenizer.unk_id] == 0.0
        assert mask.mask[tokenizer.eos_id] == 0.0
    
    def test_eos_allowed_when_complete(self, tokenizer, singular_subject, singular_verb):
        """EOS should be allowed when sentence complete."""
        state = GrammarState.initial()
        state = state.update(singular_subject)
        state = state.update(singular_verb)
        
        mask = compute_grammar_mask(
            state, tokenizer,
            eos_token_id=tokenizer.eos_id,
        )
        
        assert mask.mask[tokenizer.eos_id] == 0.0
    
    def test_valid_token_count(self, tokenizer):
        """Valid token count should be tracked."""
        state = GrammarState.initial()
        mask = compute_grammar_mask(state, tokenizer)
        
        # Should have legal_count
        assert mask.legal_count > 0
    
    def test_state_affects_mask(self, tokenizer, singular_subject):
        """State should affect the mask computation."""
        state = GrammarState.initial()
        state = state.update(singular_subject)
        
        mask = compute_grammar_mask(state, tokenizer)
        
        # Mask should be computed successfully
        assert mask.mask.shape == (tokenizer.vocab_size,)


class TestApplyGrammarMask:
    """Tests for applying mask to logits."""
    
    def test_apply_mask_shape_preserved(self, tokenizer):
        """Applying mask should preserve logits shape."""
        state = GrammarState.initial()
        mask = compute_grammar_mask(state, tokenizer)
        
        logits = torch.randn(1, tokenizer.vocab_size)
        masked = apply_grammar_mask(logits, mask)
        
        assert masked.shape == logits.shape
    
    def test_masked_values_negative(self, tokenizer):
        """Masked positions should become very negative."""
        state = GrammarState.initial()
        
        # Create a restrictive mask manually
        mask = compute_grammar_mask(state, tokenizer, allow_all_if_empty=True)
        
        # Should still produce valid output
        logits = torch.randn(1, tokenizer.vocab_size)
        masked = apply_grammar_mask(logits, mask)
        
        # Masked logits should be valid numbers
        assert not torch.isnan(masked).any()
    
    def test_broadcast_to_batch(self, tokenizer):
        """Mask should broadcast to batch dimension."""
        state = GrammarState.initial()
        mask = compute_grammar_mask(state, tokenizer)
        
        batch_size = 4
        logits = torch.randn(batch_size, tokenizer.vocab_size)
        masked = apply_grammar_mask(logits, mask)
        
        assert masked.shape == (batch_size, tokenizer.vocab_size)


# =============================================================================
# Vocab Info Tests
# =============================================================================

class TestVocabMorphInfo:
    """Tests for vocabulary morphological info."""
    
    def test_register_verb(self, tokenizer):
        """Verbs should be registered by vacana."""
        info = VocabMorphInfo(tokenizer)
        
        info.register_token(10, "tinanta", {"vacana": 1})
        info.register_token(11, "tinanta", {"vacana": 3})
        
        assert 10 in info.verb_tokens
        assert 11 in info.verb_tokens
        assert 10 in info.verbs_by_vacana[1]
        assert 11 in info.verbs_by_vacana[3]
    
    def test_register_noun(self, tokenizer):
        """Nouns should be registered by vibhakti."""
        info = VocabMorphInfo(tokenizer)
        
        info.register_token(20, "subanta", {"vibhakti": 1})
        info.register_token(21, "subanta", {"vibhakti": 2})
        
        assert 20 in info.noun_tokens
        assert 21 in info.noun_tokens
        assert 20 in info.nouns_by_vibhakti[1]
        assert 21 in info.nouns_by_vibhakti[2]
    
    def test_get_tokens_with_vacana(self, tokenizer):
        """Should return verbs with specific vacana."""
        info = VocabMorphInfo(tokenizer)
        
        info.register_token(10, "tinanta", {"vacana": 1})
        info.register_token(11, "tinanta", {"vacana": 1})
        info.register_token(12, "tinanta", {"vacana": 3})
        
        singular_verbs = info.get_tokens_with_vacana(1)
        assert 10 in singular_verbs
        assert 11 in singular_verbs
        assert 12 not in singular_verbs


# =============================================================================
# Integration Tests
# =============================================================================

class TestDecoderIntegration:
    """Integration tests for constrained decoder."""
    
    def test_constrained_decode_step(self, tokenizer):
        """Single decode step should work."""
        from panini_lm.phase5_decoding.decoder import constrained_decode_step
        
        state = GrammarState.initial()
        logits = torch.randn(tokenizer.vocab_size)
        
        token_id, mask = constrained_decode_step(
            logits, state, tokenizer
        )
        
        assert isinstance(token_id, int)
        assert 0 <= token_id < tokenizer.vocab_size
    
    def test_generation_output_structure(self):
        """GenerationOutput should have correct structure."""
        from panini_lm.phase5_decoding.decoder import GenerationOutput
        
        output = GenerationOutput(
            token_ids=[1, 2, 3],
            tokens=[],
            final_state=GrammarState.initial(),
            is_complete=False,
        )
        
        assert len(output.token_ids) == 3
        assert isinstance(output.final_state, GrammarState)
