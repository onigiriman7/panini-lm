"""
Tests for Phase 4: Semantic Maturation FFN Layer.

Test categories:
1. FFN tests — basic feed-forward network
2. Gated FFN tests — SwiGLU-style gated FFN
3. Block tests — combined attention + FFN
4. Stack tests — multiple layers

Run with: pytest tests/unit/test_phase4.py -v
"""

import pytest
import torch
import torch.nn as nn

from panini_lm.core.types import AdjacencyMatrix
from panini_lm.core.config import AttentionConfig, FFNConfig
from panini_lm.phase4_ffn import PaniniFeedForward, TransformerBlock
from panini_lm.phase4_ffn.ffn import GatedFeedForward
from panini_lm.phase4_ffn.block import TransformerStack


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def batch_input():
    """Standard batch input."""
    batch, seq_len, d_model = 2, 8, 64
    return torch.randn(batch, seq_len, d_model)


@pytest.fixture
def adjacency() -> AdjacencyMatrix:
    """Fully connected adjacency for testing."""
    seq_len = 8
    return AdjacencyMatrix(
        matrix=torch.zeros(seq_len, seq_len),
        meta={"seq_len": seq_len},
        links=[],
    )


# =============================================================================
# FFN Tests
# =============================================================================

class TestPaniniFeedForward:
    """Tests for standard FFN."""
    
    def test_output_shape(self, batch_input):
        """FFN should preserve shape."""
        d_model = batch_input.shape[-1]
        ffn = PaniniFeedForward(d_model=d_model)
        
        output = ffn(batch_input)
        assert output.shape == batch_input.shape
    
    def test_expansion_factor(self):
        """Hidden dimension should match expansion factor."""
        d_model = 64
        expansion = 2.0
        ffn = PaniniFeedForward(d_model=d_model, expansion_factor=expansion)
        
        assert ffn.d_hidden == 128  # 64 * 2.0
    
    def test_default_expansion_1_5x(self):
        """Default expansion should be 1.5x."""
        d_model = 64
        ffn = PaniniFeedForward(d_model=d_model)
        
        assert ffn.expansion_factor == 1.5
        assert ffn.d_hidden == 96  # 64 * 1.5
    
    def test_parameter_count(self):
        """Check parameter count matches expected."""
        d_model = 64
        ffn = PaniniFeedForward(d_model=d_model, expansion_factor=1.5)
        
        # W1: d_model * d_hidden + d_hidden (bias)
        # W2: d_hidden * d_model + d_model (bias)
        d_hidden = 96
        expected = (d_model * d_hidden + d_hidden) + (d_hidden * d_model + d_model)
        actual = sum(p.numel() for p in ffn.parameters())
        
        assert actual == expected
    
    def test_no_bias_option(self):
        """Should support no-bias configuration."""
        ffn = PaniniFeedForward(d_model=64, bias=False)
        
        assert ffn.W1.bias is None
        assert ffn.W2.bias is None
    
    def test_different_activations(self, batch_input):
        """Different activations should work."""
        d_model = batch_input.shape[-1]
        
        for activation in ["gelu", "relu", "silu"]:
            ffn = PaniniFeedForward(d_model=d_model, activation=activation)
            output = ffn(batch_input)
            assert output.shape == batch_input.shape
    
    def test_deterministic_in_eval(self, batch_input):
        """FFN should be deterministic in eval mode."""
        d_model = batch_input.shape[-1]
        ffn = PaniniFeedForward(d_model=d_model, dropout=0.0)
        ffn.eval()
        
        out1 = ffn(batch_input)
        out2 = ffn(batch_input)
        
        assert torch.allclose(out1, out2)


class TestGatedFeedForward:
    """Tests for gated (SwiGLU-style) FFN."""
    
    def test_output_shape(self, batch_input):
        """Gated FFN should preserve shape."""
        d_model = batch_input.shape[-1]
        ffn = GatedFeedForward(d_model=d_model)
        
        output = ffn(batch_input)
        assert output.shape == batch_input.shape
    
    def test_gated_has_three_projections(self):
        """Gated FFN should have gate, up, and down."""
        ffn = GatedFeedForward(d_model=64)
        
        assert hasattr(ffn, 'W_gate')
        assert hasattr(ffn, 'W_up')
        assert hasattr(ffn, 'W_down')
    
    def test_hidden_dim_adjusted(self):
        """Hidden dim should be adjusted for gating."""
        d_model = 64
        expansion = 1.5
        ffn = GatedFeedForward(d_model=d_model, expansion_factor=expansion)
        
        # Gated uses 2/3 of expansion to match parameter count
        expected_hidden = int(d_model * expansion * 2 / 3)
        assert ffn.d_hidden == expected_hidden


# =============================================================================
# Transformer Block Tests
# =============================================================================

class TestTransformerBlock:
    """Tests for combined attention + FFN block."""
    
    def test_output_shape(self, batch_input, adjacency):
        """Block should preserve shape."""
        d_model = batch_input.shape[-1]
        block = TransformerBlock(d_model=d_model, n_heads=4)
        
        output = block(batch_input, adjacency)
        assert output.shape == batch_input.shape
    
    def test_block_creation(self):
        """Block should be creatable with various configs."""
        block = TransformerBlock(
            d_model=128,
            n_heads=8,
            expansion_factor=2.0,
            dropout=0.1,
            attention_dropout=0.1,
        )
        
        assert block.d_model == 128
        assert block.n_heads == 8
    
    def test_has_all_components(self):
        """Block should have all required components."""
        block = TransformerBlock(d_model=64, n_heads=4)
        
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
        assert hasattr(block, 'attention')
        assert hasattr(block, 'ffn')
        assert hasattr(block, 'qkv_projection')
    
    def test_from_configs(self):
        """Block creation from configs."""
        attn_config = AttentionConfig(attention_dropout=0.2)
        ffn_config = FFNConfig(expansion_factor=1.5)
        
        block = TransformerBlock.from_configs(
            attention_config=attn_config,
            ffn_config=ffn_config,
            d_model=64,
            n_heads=4,
        )
        
        assert block.d_model == 64
    
    def test_residual_connection(self, adjacency):
        """Output should be influenced by residual."""
        d_model = 64
        block = TransformerBlock(d_model=d_model, n_heads=4, dropout=0.0)
        block.eval()
        
        # With identity-like output, residual should dominate
        x = torch.randn(1, 8, d_model)
        output = block(x, adjacency)
        
        # Output should be different from input (due to FFN)
        # but not completely different (due to residual)
        # This is a weak test but ensures residual is connected
        assert not torch.allclose(x, output)
    
    def test_gated_ffn_option(self, batch_input, adjacency):
        """Block should support gated FFN."""
        d_model = batch_input.shape[-1]
        block = TransformerBlock(
            d_model=d_model,
            n_heads=4,
            ffn_type="gated",
        )
        
        output = block(batch_input, adjacency)
        assert output.shape == batch_input.shape


# =============================================================================
# Transformer Stack Tests
# =============================================================================

class TestTransformerStack:
    """Tests for stacked transformer layers."""
    
    def test_output_shape(self, batch_input, adjacency):
        """Stack should preserve shape."""
        d_model = batch_input.shape[-1]
        stack = TransformerStack(
            n_layers=3,
            d_model=d_model,
            n_heads=4,
        )
        
        output = stack(batch_input, adjacency)
        assert output.shape == batch_input.shape
    
    def test_correct_number_of_layers(self):
        """Stack should have correct number of layers."""
        stack = TransformerStack(n_layers=6, d_model=64, n_heads=4)
        
        assert len(stack.layers) == 6
    
    def test_has_final_norm(self):
        """Stack should have final layer norm."""
        stack = TransformerStack(n_layers=2, d_model=64, n_heads=4)
        
        assert hasattr(stack, 'final_norm')
        assert isinstance(stack.final_norm, nn.LayerNorm)
    
    def test_parameter_scaling(self):
        """More layers should have more parameters."""
        stack2 = TransformerStack(n_layers=2, d_model=64, n_heads=4)
        stack4 = TransformerStack(n_layers=4, d_model=64, n_heads=4)
        
        params2 = sum(p.numel() for p in stack2.parameters())
        params4 = sum(p.numel() for p in stack4.parameters())
        
        # 4-layer should have roughly 2x params (minus final norm)
        assert params4 > params2 * 1.5


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhaseIntegration:
    """Test Phase 4 integration with earlier phases."""
    
    def test_with_phase2_outputs(self):
        """Block should work with Phase 2A + 2B outputs."""
        from panini_lm.phase2a_symbolic import build_adjacency_matrix
        from panini_lm.phase2b_neural import process_neural
        from panini_lm.core.config import NeuralConfig
        
        # Tokens and Phase 2A
        tokens = [
            {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta",
             "attributes": {"vibhakti": 1, "vacana": 1}},
            {"surface": "gacchati", "stem": "gam", "type": "tinanta",
             "attributes": {"vacana": 1}},
        ]
        adjacency = build_adjacency_matrix(tokens)
        
        # Phase 2B
        config = NeuralConfig(d_model=64, num_heads=4)
        phase2b = process_neural(tokens, config=config)
        embeddings = phase2b["embeddings"]
        
        # Phase 4 block
        block = TransformerBlock(d_model=64, n_heads=4)
        block.eval()
        
        output = block(embeddings, adjacency)
        assert output.shape == embeddings.shape
    
    def test_full_forward_pass(self):
        """Test complete forward through stack."""
        from panini_lm.phase2a_symbolic import build_adjacency_matrix
        
        tokens = [
            {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta",
             "attributes": {"vibhakti": 1, "vacana": 1}},
            {"surface": "gṛham", "stem": "gṛha", "type": "subanta",
             "attributes": {"vibhakti": 2, "vacana": 1}},
            {"surface": "gacchati", "stem": "gam", "type": "tinanta",
             "attributes": {"vacana": 1}},
        ]
        adjacency = build_adjacency_matrix(tokens)
        
        # Create embeddings
        d_model = 64
        embeddings = torch.randn(1, len(tokens), d_model)
        
        # Stack
        stack = TransformerStack(
            n_layers=4,
            d_model=d_model,
            n_heads=4,
            expansion_factor=1.5,
        )
        stack.eval()
        
        output = stack(embeddings, adjacency)
        
        assert output.shape == embeddings.shape
        assert not torch.isnan(output).any()
