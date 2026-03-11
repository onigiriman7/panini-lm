"""Tests for configuration classes."""

import pytest

from panini_lm.core.config import (
    PaniniConfig,
    MorphologyConfig,
    SymbolicConfig,
    NeuralConfig,
    AttentionConfig,
    FFNConfig,
    DecodingConfig,
    get_small_config,
    get_base_config,
    get_large_config,
)


class TestPaniniConfig:
    """Tests for master PaniniConfig."""
    
    def test_default_config(self):
        """Create config with all defaults."""
        config = PaniniConfig()
        
        assert config.num_layers == 6
        assert config.neural.d_model == 512
        assert config.neural.num_heads == 8
        assert config.ffn.expansion_factor == 1.5
    
    def test_computed_properties(self):
        """Test computed properties."""
        config = PaniniConfig()
        
        assert config.head_dim == 512 // 8  # 64
        assert config.d_ff == int(512 * 1.5)  # 768
    
    def test_custom_config(self):
        """Create config with custom values."""
        config = PaniniConfig(
            neural=NeuralConfig(d_model=768, num_heads=12),
            ffn=FFNConfig(expansion_factor=2.0),
            num_layers=12,
        )
        
        assert config.neural.d_model == 768
        assert config.neural.num_heads == 12
        assert config.head_dim == 64
        assert config.ffn.expansion_factor == 2.0
        assert config.d_ff == 768 * 2
    
    def test_validation_passes(self):
        """Valid config should pass validation."""
        config = PaniniConfig()
        config.validate()  # Should not raise
    
    def test_validation_fails_head_mismatch(self):
        """Invalid head dimension should fail validation."""
        config = PaniniConfig(
            neural=NeuralConfig(d_model=512, num_heads=7)  # 512 not divisible by 7
        )
        
        with pytest.raises(AssertionError, match="divisible"):
            config.validate()
    
    def test_config_immutability(self):
        """Config should be frozen (immutable)."""
        config = PaniniConfig()
        
        with pytest.raises(Exception):  # FrozenInstanceError
            config.num_layers = 12


class TestMorphologyConfig:
    """Tests for MorphologyConfig."""
    
    def test_defaults(self):
        """Check default values."""
        config = MorphologyConfig()
        
        assert config.backend == "auto"
        assert config.enable_samasa is True
        assert config.normalize_unicode is True
        assert config.max_sandhi_candidates == 5
    
    def test_custom_backend(self):
        """Set specific backend."""
        config = MorphologyConfig(backend="vidyut")
        assert config.backend == "vidyut"


class TestNeuralConfig:
    """Tests for NeuralConfig."""
    
    def test_no_positional_encoding_default(self):
        """Position encoding should be OFF by default (free word order!)."""
        config = NeuralConfig()
        assert config.use_positional_encoding is False
    
    def test_model_dimensions(self):
        """Test dimension settings."""
        config = NeuralConfig(
            vocab_size=100000,
            d_model=1024,
            num_heads=16,
        )
        assert config.vocab_size == 100000
        assert config.d_model == 1024


class TestFFNConfig:
    """Tests for FFNConfig."""
    
    def test_reduced_expansion(self):
        """FFN expansion should be 1.5x (not standard 4x)."""
        config = FFNConfig()
        assert config.expansion_factor == 1.5
        assert config.expansion_factor < 4  # Critical design choice!
    
    def test_swiglu_default(self):
        """SwiGLU should be default activation."""
        config = FFNConfig()
        assert config.activation == "swiglu"


class TestDecodingConfig:
    """Tests for DecodingConfig."""
    
    def test_grammar_enforcement_default(self):
        """Grammar enforcement should be ON by default."""
        config = DecodingConfig()
        assert config.enforce_grammar is True
    
    def test_generation_params(self):
        """Test generation parameters."""
        config = DecodingConfig(
            max_length=256,
            temperature=0.8,
            top_k=50,
        )
        assert config.max_length == 256
        assert config.temperature == 0.8
        assert config.top_k == 50


class TestPresetConfigs:
    """Tests for preset configurations."""
    
    def test_small_config(self):
        """Small config for testing."""
        config = get_small_config()
        
        assert config.neural.d_model == 256
        assert config.neural.num_heads == 4
        assert config.num_layers == 4
        config.validate()  # Should pass
    
    def test_base_config(self):
        """Base config (default)."""
        config = get_base_config()
        
        assert config.neural.d_model == 512
        assert config.num_layers == 6
        config.validate()
    
    def test_large_config(self):
        """Large config for production."""
        config = get_large_config()
        
        assert config.neural.d_model == 1024
        assert config.neural.num_heads == 16
        assert config.num_layers == 12
        config.validate()
    
    def test_all_configs_valid(self):
        """All preset configs should be valid."""
        configs = [get_small_config(), get_base_config(), get_large_config()]
        
        for config in configs:
            config.validate()  # Should not raise
