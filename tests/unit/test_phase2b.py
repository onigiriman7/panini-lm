"""
Tests for Phase 2B: Neural Engine.

Test categories:
1. Tokenizer tests — vocabulary management
2. Embedding tests — token + type + feature embedding
3. Projection tests — Q, K, V computation
4. Pipeline tests — end-to-end processing

Run with: pytest tests/unit/test_phase2b.py -v
"""

import pytest
import torch

from panini_lm.core.types import MorphToken
from panini_lm.core.config import NeuralConfig
from panini_lm.phase2b_neural import (
    PaniniTokenizer,
    MorphAwareEmbedding,
    QKVProjection,
    process_neural,
)
from panini_lm.phase2b_neural.processor import NeuralPipeline


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_config() -> NeuralConfig:
    """Small config for testing."""
    return NeuralConfig(
        d_model=64,
        num_heads=4,
        vocab_size=1000,
        embedding_dropout=0.0,  # Deterministic for tests
        use_positional_encoding=False,
    )


@pytest.fixture
def tokenizer() -> PaniniTokenizer:
    """Pre-built tokenizer with some vocab."""
    tok = PaniniTokenizer()
    # Add some stems
    tok.add_token("rāma")
    tok.add_token("gam")
    tok.add_token("gṛha")
    return tok


@pytest.fixture
def sample_tokens() -> list[MorphToken]:
    """Sample sentence tokens."""
    return [
        {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta",
         "attributes": {"vibhakti": 1, "vacana": 1, "linga": "m"}},
        {"surface": "gṛham", "stem": "gṛha", "type": "subanta",
         "attributes": {"vibhakti": 2, "vacana": 1, "linga": "n"}},
        {"surface": "gacchati", "stem": "gam", "type": "tinanta",
         "attributes": {"vacana": 1, "lakara": "lat"}},
    ]


# =============================================================================
# Tokenizer Tests
# =============================================================================

class TestPaniniTokenizer:
    """Tests for tokenizer."""
    
    def test_special_tokens_initialized(self):
        """Special tokens should be pre-loaded."""
        tok = PaniniTokenizer()
        
        assert "[PAD]" in tok.vocab
        assert "[UNK]" in tok.vocab
        assert "[BOS]" in tok.vocab
        assert "[EOS]" in tok.vocab
        assert "[MASK]" in tok.vocab
    
    def test_special_token_ids(self):
        """Special tokens should have expected IDs."""
        tok = PaniniTokenizer()
        
        assert tok.pad_id == 0
        assert tok.unk_id == 1
        assert tok.bos_id == 2
        assert tok.eos_id == 3
        assert tok.mask_id == 4
    
    def test_add_token(self):
        """Adding token should increase vocab size."""
        tok = PaniniTokenizer()
        initial = tok.vocab_size
        
        tok.add_token("test")
        assert tok.vocab_size == initial + 1
        assert "test" in tok.vocab
    
    def test_add_duplicate_token(self):
        """Adding duplicate should not increase size."""
        tok = PaniniTokenizer()
        tok.add_token("test")
        size = tok.vocab_size
        
        tok.add_token("test")
        assert tok.vocab_size == size
    
    def test_encode_known_stem(self, tokenizer, sample_tokens):
        """Encoding known stem should return correct ID."""
        token = sample_tokens[0]  # "rāma"
        
        tid = tokenizer.encode_token(token)
        assert tid == tokenizer.vocab["rāma"]
    
    def test_encode_unknown_returns_unk(self, tokenizer):
        """Unknown token should return UNK ID."""
        unknown: MorphToken = {
            "surface": "xyz",
            "stem": "xyz",
            "type": "unknown",
            "attributes": {}
        }
        
        tid = tokenizer.encode_token(unknown)
        assert tid == tokenizer.unk_id
    
    def test_encode_sequence(self, tokenizer, sample_tokens):
        """Encode full sequence."""
        token_ids, type_ids = tokenizer.encode(sample_tokens, add_bos=False, add_eos=False)
        
        assert len(token_ids) == 3
        assert len(type_ids) == 3
    
    def test_encode_with_bos_eos(self, tokenizer, sample_tokens):
        """BOS/EOS should be added."""
        token_ids, type_ids = tokenizer.encode(sample_tokens, add_bos=True, add_eos=True)
        
        assert len(token_ids) == 5  # 3 + BOS + EOS
        assert token_ids[0] == tokenizer.bos_id
        assert token_ids[-1] == tokenizer.eos_id
    
    def test_encode_type(self, tokenizer, sample_tokens):
        """Type encoding should work."""
        _, type_ids = tokenizer.encode(sample_tokens, add_bos=False, add_eos=False)
        
        # First two are subanta (0), third is tinanta (1)
        assert type_ids[0] == 0  # subanta
        assert type_ids[1] == 0  # subanta
        assert type_ids[2] == 1  # tinanta
    
    def test_decode(self, tokenizer):
        """Decoding should return token strings."""
        ids = [tokenizer.vocab["rāma"], tokenizer.vocab["gam"]]
        decoded = tokenizer.decode(ids)
        
        assert decoded == ["rāma", "gam"]
    
    def test_type_vocab(self, tokenizer):
        """Type vocabulary should be initialized."""
        assert "subanta" in tokenizer.type_vocab
        assert "tinanta" in tokenizer.type_vocab
        assert "avyaya" in tokenizer.type_vocab


# =============================================================================
# Embedding Tests
# =============================================================================

class TestMorphAwareEmbedding:
    """Tests for embedding layer."""
    
    def test_output_shape(self, basic_config):
        """Embedding output should have correct shape."""
        emb = MorphAwareEmbedding.from_config(basic_config, vocab_size=100)
        
        batch, seq = 2, 5
        token_ids = torch.randint(0, 100, (batch, seq))
        type_ids = torch.randint(0, 7, (batch, seq))
        
        output = emb(token_ids, type_ids)
        assert output.shape == (batch, seq, basic_config.d_model)
    
    def test_no_positional_encoding(self, basic_config):
        """Different positions should get same embedding for same token."""
        emb = MorphAwareEmbedding.from_config(basic_config, vocab_size=100)
        
        # Same token at different positions
        token_ids = torch.tensor([[5, 5, 5]])  # Same token
        type_ids = torch.tensor([[0, 0, 0]])   # Same type
        
        output = emb(token_ids, type_ids)
        
        # All positions should produce same embedding (modulo dropout in training)
        emb.eval()  # Disable dropout
        output = emb(token_ids, type_ids)
        
        # Check embeddings are equal (no positional bias)
        assert torch.allclose(output[0, 0], output[0, 1], atol=1e-5)
        assert torch.allclose(output[0, 1], output[0, 2], atol=1e-5)
    
    def test_padding_zeroed(self, basic_config):
        """Padding token should have near-zero embedding."""
        emb = MorphAwareEmbedding.from_config(basic_config, vocab_size=100)
        
        # Get embedding for padding (ID 0)
        token_ids = torch.tensor([[0]])  # Padding
        type_ids = torch.tensor([[0]])
        
        # Raw token embedding should be zero
        raw_emb = emb.token_embedding(token_ids)
        assert torch.allclose(raw_emb, torch.zeros_like(raw_emb))
    
    def test_with_features(self, basic_config):
        """Feature embeddings should be added when enabled."""
        emb = MorphAwareEmbedding(
            vocab_size=100,
            d_model=basic_config.d_model,
            combine_features=True,
        )
        
        batch, seq = 1, 3
        token_ids = torch.randint(1, 100, (batch, seq))
        type_ids = torch.randint(0, 7, (batch, seq))
        
        feature_ids = {
            "vibhakti": torch.tensor([[1, 2, 0]]),
            "vacana": torch.tensor([[1, 1, 1]]),
            "linga": torch.tensor([[1, 3, 0]]),
            "lakara": torch.tensor([[0, 0, 1]]),
        }
        
        output = emb(token_ids, type_ids, feature_ids)
        assert output.shape == (batch, seq, basic_config.d_model)
    
    def test_different_types_different_embeddings(self, basic_config):
        """Different types should produce different embeddings."""
        emb = MorphAwareEmbedding.from_config(basic_config, vocab_size=100)
        emb.eval()
        
        # Same token, different types
        token_ids = torch.tensor([[5, 5]])
        type_ids = torch.tensor([[0, 1]])  # subanta vs tinanta
        
        output = emb(token_ids, type_ids)
        
        # Should NOT be equal
        assert not torch.allclose(output[0, 0], output[0, 1])


# =============================================================================
# Projection Tests
# =============================================================================

class TestQKVProjection:
    """Tests for Q, K, V projection."""
    
    def test_output_shapes(self, basic_config):
        """Q, K, V should have correct shapes."""
        proj = QKVProjection.from_config(basic_config)
        
        batch, seq = 2, 5
        x = torch.randn(batch, seq, basic_config.d_model)
        
        Q, K, V = proj(x)
        
        d_head = basic_config.d_model // basic_config.num_heads
        expected_shape = (batch, basic_config.num_heads, seq, d_head)
        
        assert Q.shape == expected_shape
        assert K.shape == expected_shape
        assert V.shape == expected_shape
    
    def test_return_dict(self, basic_config):
        """Return dict should contain all required fields."""
        proj = QKVProjection.from_config(basic_config)
        
        x = torch.randn(1, 3, basic_config.d_model)
        qkv = proj(x, return_dict=True)
        
        assert "Q" in qkv
        assert "K" in qkv
        assert "V" in qkv
        assert "d_head" in qkv
        assert "n_heads" in qkv
    
    def test_output_projection(self, basic_config):
        """Output projection should restore d_model dimension."""
        proj = QKVProjection.from_config(basic_config)
        
        batch, seq = 2, 5
        d_head = basic_config.d_model // basic_config.num_heads
        
        # Simulate attention output
        attn_out = torch.randn(batch, basic_config.num_heads, seq, d_head)
        
        output = proj.project_output(attn_out)
        assert output.shape == (batch, seq, basic_config.d_model)
    
    def test_deterministic_in_eval(self, basic_config):
        """Projection should be deterministic in eval mode."""
        proj = QKVProjection.from_config(basic_config)
        proj.eval()
        
        x = torch.randn(1, 3, basic_config.d_model)
        
        Q1, K1, V1 = proj(x)
        Q2, K2, V2 = proj(x)
        
        assert torch.allclose(Q1, Q2)
        assert torch.allclose(K1, K2)
        assert torch.allclose(V1, V2)


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestNeuralPipeline:
    """Tests for full pipeline."""
    
    def test_pipeline_creation(self, basic_config, tokenizer):
        """Pipeline should be creatable from config."""
        pipeline = NeuralPipeline.from_config(
            config=basic_config,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        
        assert pipeline.tokenizer is tokenizer
        assert pipeline.embedding is not None
        assert pipeline.projection is not None
    
    def test_process_tokens(self, basic_config, tokenizer, sample_tokens):
        """Processing should produce valid output."""
        pipeline = NeuralPipeline.from_config(
            config=basic_config,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        
        result = pipeline.process(sample_tokens)
        
        assert "embeddings" in result
        assert "qkv" in result
        assert "token_ids" in result
        assert result["seq_len"] == 3
    
    def test_embeddings_shape(self, basic_config, tokenizer, sample_tokens):
        """Embeddings should have correct shape."""
        pipeline = NeuralPipeline.from_config(
            config=basic_config,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        
        result = pipeline.process(sample_tokens)
        
        # (batch=1, seq=3, d_model)
        assert result["embeddings"].shape == (1, 3, basic_config.d_model)
    
    def test_qkv_shapes(self, basic_config, tokenizer, sample_tokens):
        """QKV should have correct shapes."""
        pipeline = NeuralPipeline.from_config(
            config=basic_config,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
        )
        
        result = pipeline.process(sample_tokens)
        qkv = result["qkv"]
        
        d_head = basic_config.d_model // basic_config.num_heads
        expected = (1, basic_config.num_heads, 3, d_head)
        
        assert qkv["Q"].shape == expected
        assert qkv["K"].shape == expected
        assert qkv["V"].shape == expected


class TestProcessNeural:
    """Tests for module-level process function."""
    
    def test_process_neural_basic(self, sample_tokens):
        """process_neural should work with defaults."""
        result = process_neural(sample_tokens)
        
        assert "embeddings" in result
        assert "qkv" in result
    
    def test_process_neural_with_config(self, basic_config, sample_tokens):
        """process_neural should accept config."""
        result = process_neural(
            tokens=sample_tokens,
            config=basic_config,
        )
        
        assert result["embeddings"].shape[-1] == basic_config.d_model
    
    def test_empty_tokens_raises(self):
        """Empty token list should raise error."""
        from panini_lm.core.exceptions import NeuralEngineError
        
        with pytest.raises(NeuralEngineError):
            process_neural([])


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase1To2BIntegration:
    """Test Phase 1 → Phase 2B connection."""
    
    def test_phase1_output_compatible(self, basic_config, tokenizer):
        """Phase 1 output format should work with Phase 2B."""
        # Simulate Phase 1 output
        phase1_output = {
            "tokens": [
                {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta",
                 "attributes": {"vibhakti": 1, "vacana": 1}},
            ],
            "sandhi_splits": ["rāmaḥ"],
            "confidence": 1.0,
        }
        
        # Process through Phase 2B
        result = process_neural(
            tokens=phase1_output["tokens"],
            config=basic_config,
            tokenizer=tokenizer,
        )
        
        assert result["seq_len"] == 1
