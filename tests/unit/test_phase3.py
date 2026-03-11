"""
Tests for Phase 3: Sparse Grammar-Constrained Attention.

Test categories:
1. Backend tests — selection and availability
2. Attention tests — core computation
3. Masking tests — verify grammar constraints applied
4. Integration tests — Phase 2A + 2B → Phase 3

Run with: pytest tests/unit/test_phase3.py -v
"""

import pytest
import torch
import math

from panini_lm.core.types import AdjacencyMatrix
from panini_lm.core.config import AttentionConfig
from panini_lm.phase3_attention import (
    sparse_attention,
    SparseAttentionLayer,
    select_backend,
    AttentionBackend,
)
from panini_lm.phase3_attention.backend import is_triton_available


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_qkv():
    """Basic Q, K, V tensors for testing."""
    batch, n_heads, seq_len, d_head = 2, 4, 8, 16
    
    Q = torch.randn(batch, n_heads, seq_len, d_head)
    K = torch.randn(batch, n_heads, seq_len, d_head)
    V = torch.randn(batch, n_heads, seq_len, d_head)
    
    return Q, K, V


@pytest.fixture
def full_adjacency() -> AdjacencyMatrix:
    """Fully connected adjacency matrix (no masking)."""
    seq_len = 8
    matrix = torch.zeros(seq_len, seq_len)
    
    return AdjacencyMatrix(
        matrix=matrix,
        meta={"seq_len": seq_len, "sparsity_ratio": 1.0},
        links=[],
    )


@pytest.fixture
def diagonal_adjacency() -> AdjacencyMatrix:
    """Self-attention only adjacency (extremely sparse)."""
    seq_len = 8
    matrix = torch.full((seq_len, seq_len), float('-inf'))
    for i in range(seq_len):
        matrix[i, i] = 0.0
    
    return AdjacencyMatrix(
        matrix=matrix,
        meta={"seq_len": seq_len, "sparsity_ratio": seq_len / (seq_len * seq_len)},
        links=[{"source_idx": i, "target_idx": i} for i in range(seq_len)],
    )


@pytest.fixture
def sparse_adjacency() -> AdjacencyMatrix:
    """Typical sparse adjacency (subject-verb, object-verb)."""
    seq_len = 8
    matrix = torch.full((seq_len, seq_len), float('-inf'))
    
    # Self-attention
    for i in range(seq_len):
        matrix[i, i] = 0.0
    
    # Add some connections (simulating S-V-O patterns)
    # Position 0 (subject) → 2 (verb)
    matrix[0, 2] = 0.0
    matrix[2, 0] = 0.0
    
    # Position 1 (object) → 2 (verb)
    matrix[1, 2] = 0.0
    matrix[2, 1] = 0.0
    
    return AdjacencyMatrix(
        matrix=matrix,
        meta={"seq_len": seq_len},
        links=[],
    )


# =============================================================================
# Backend Tests
# =============================================================================

class TestBackendSelection:
    """Tests for backend selection."""
    
    def test_pytorch_always_available(self):
        """PyTorch backend should always be selectable."""
        backend = select_backend("pytorch")
        assert backend == AttentionBackend.PYTORCH
    
    def test_auto_selects_something(self):
        """Auto mode should select a valid backend."""
        backend = select_backend("auto")
        assert backend in [AttentionBackend.PYTORCH, AttentionBackend.TRITON]
    
    def test_triton_unavailable_raises(self):
        """If Triton unavailable and requested, should raise."""
        if is_triton_available():
            pytest.skip("Triton is available, can't test unavailable case")
        
        with pytest.raises(RuntimeError):
            select_backend("triton")


# =============================================================================
# Attention Computation Tests
# =============================================================================

class TestSparseAttention:
    """Tests for sparse attention computation."""
    
    def test_output_shape(self, basic_qkv, full_adjacency):
        """Output should have same shape as Q, K, V."""
        Q, K, V = basic_qkv
        
        output = sparse_attention(Q, K, V, full_adjacency, training=False)
        
        assert output.shape == Q.shape
    
    def test_deterministic_in_eval(self, basic_qkv, full_adjacency):
        """Same input should produce same output when not training."""
        Q, K, V = basic_qkv
        
        out1 = sparse_attention(Q, K, V, full_adjacency, training=False)
        out2 = sparse_attention(Q, K, V, full_adjacency, training=False)
        
        assert torch.allclose(out1, out2)
    
    def test_batch_independence(self, full_adjacency):
        """Each batch should be processed independently."""
        n_heads, seq_len, d_head = 2, 4, 8
        
        Q1 = torch.randn(1, n_heads, seq_len, d_head)
        K1 = torch.randn(1, n_heads, seq_len, d_head)
        V1 = torch.randn(1, n_heads, seq_len, d_head)
        
        Q2 = torch.randn(1, n_heads, seq_len, d_head)
        K2 = torch.randn(1, n_heads, seq_len, d_head)
        V2 = torch.randn(1, n_heads, seq_len, d_head)
        
        # Smaller adjacency for this test
        adj = AdjacencyMatrix(
            matrix=torch.zeros(seq_len, seq_len),
            meta={"seq_len": seq_len},
            links=[],
        )
        
        # Process separately
        out1 = sparse_attention(Q1, K1, V1, adj, training=False)
        out2 = sparse_attention(Q2, K2, V2, adj, training=False)
        
        # Process together
        Q_batch = torch.cat([Q1, Q2], dim=0)
        K_batch = torch.cat([K1, K2], dim=0)
        V_batch = torch.cat([V1, V2], dim=0)
        
        out_batch = sparse_attention(Q_batch, K_batch, V_batch, adj, training=False)
        
        # Results should match
        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)


class TestMaskingBehavior:
    """Tests for grammar masking."""
    
    def test_full_connectivity_no_masking(self, basic_qkv, full_adjacency):
        """Full adjacency should produce standard attention."""
        Q, K, V = basic_qkv
        
        output = sparse_attention(Q, K, V, full_adjacency, training=False)
        
        # Should have valid numbers (not NaN or Inf)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_diagonal_only_self_values(self, diagonal_adjacency):
        """Diagonal adjacency should only attend to self."""
        n_heads, seq_len, d_head = 2, 8, 16
        batch = 1
        
        # Create distinct V values for each position
        Q = torch.randn(batch, n_heads, seq_len, d_head)
        K = torch.randn(batch, n_heads, seq_len, d_head)
        V = torch.zeros(batch, n_heads, seq_len, d_head)
        
        # Set V[i] to be unique identifier
        for i in range(seq_len):
            V[0, :, i, :] = float(i + 1)
        
        output = sparse_attention(Q, K, V, diagonal_adjacency, training=False)
        
        # Each position should only see its own V (since only self-attention allowed)
        # After softmax on [0, -inf, -inf, ...], only position i gets weight 1.0
        for i in range(seq_len):
            expected = float(i + 1)
            assert torch.allclose(
                output[0, :, i, :],
                torch.full_like(output[0, :, i, :], expected),
                atol=1e-5
            )
    
    def test_sparse_connectivity(self, sparse_adjacency):
        """Sparse adjacency should restrict attention flow."""
        n_heads, seq_len, d_head = 2, 8, 16
        batch = 1
        
        Q = torch.randn(batch, n_heads, seq_len, d_head)
        K = torch.randn(batch, n_heads, seq_len, d_head)
        V = torch.randn(batch, n_heads, seq_len, d_head)
        
        output = sparse_attention(Q, K, V, sparse_adjacency, training=False)
        
        # Should produce valid output
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_masked_positions_dont_contribute(self):
        """Masked positions should not affect output."""
        batch, n_heads, seq_len, d_head = 1, 1, 3, 4
        
        Q = torch.randn(batch, n_heads, seq_len, d_head)
        K = torch.randn(batch, n_heads, seq_len, d_head)
        V = torch.randn(batch, n_heads, seq_len, d_head)
        
        # Position 0 can only attend to position 1
        mask = torch.full((seq_len, seq_len), float('-inf'))
        mask[0, 1] = 0.0  # Only 0 → 1 allowed
        mask[1, 1] = 0.0  # Self for position 1
        mask[2, 2] = 0.0  # Self for position 2
        
        adj = AdjacencyMatrix(matrix=mask, meta={}, links=[])
        
        output = sparse_attention(Q, K, V, adj, training=False)
        
        # Position 0's output should equal V[1] (only valid attention target)
        assert torch.allclose(output[0, 0, 0], V[0, 0, 1], atol=1e-5)


# =============================================================================
# Shape Validation Tests
# =============================================================================

class TestShapeValidation:
    """Tests for input shape validation."""
    
    def test_wrong_qkv_dimensions_raises(self, full_adjacency):
        """QKV with wrong dimensions should raise."""
        from panini_lm.core.exceptions import ShapeMismatchError
        
        Q = torch.randn(2, 4, 8)  # 3D instead of 4D
        K = torch.randn(2, 4, 8)
        V = torch.randn(2, 4, 8)
        
        with pytest.raises(ShapeMismatchError):
            sparse_attention(Q, K, V, full_adjacency)
    
    def test_mismatched_qkv_shapes_raises(self, full_adjacency):
        """Different Q, K, V shapes should raise."""
        from panini_lm.core.exceptions import ShapeMismatchError
        
        Q = torch.randn(1, 2, 8, 16)
        K = torch.randn(1, 2, 8, 16)
        V = torch.randn(1, 2, 4, 16)  # Different seq_len
        
        with pytest.raises(ShapeMismatchError):
            sparse_attention(Q, K, V, full_adjacency)
    
    def test_mismatched_mask_size_raises(self):
        """Mask size not matching seq_len should raise."""
        from panini_lm.core.exceptions import ShapeMismatchError
        
        Q = torch.randn(1, 2, 8, 16)  # seq_len = 8
        K = torch.randn(1, 2, 8, 16)
        V = torch.randn(1, 2, 8, 16)
        
        # Wrong size adjacency
        adj = AdjacencyMatrix(
            matrix=torch.zeros(4, 4),  # 4 != 8
            meta={},
            links=[],
        )
        
        with pytest.raises(ShapeMismatchError):
            sparse_attention(Q, K, V, adj)


# =============================================================================
# Layer Tests
# =============================================================================

class TestSparseAttentionLayer:
    """Tests for nn.Module wrapper."""
    
    def test_layer_creation(self):
        """Layer should be creatable."""
        layer = SparseAttentionLayer(d_model=64, n_heads=4)
        
        assert layer.d_model == 64
        assert layer.n_heads == 4
        assert layer.d_head == 16
    
    def test_layer_forward(self, basic_qkv, full_adjacency):
        """Layer forward should work."""
        layer = SparseAttentionLayer(d_model=64, n_heads=4, dropout=0.0)
        layer.eval()
        
        Q, K, V = basic_qkv
        output = layer(Q, K, V, full_adjacency)
        
        assert output.shape == Q.shape
    
    def test_layer_from_config(self):
        """Layer creation from config."""
        config = AttentionConfig(attention_dropout=0.2)
        layer = SparseAttentionLayer.from_config(config, d_model=128, n_heads=8)
        
        assert layer.dropout == 0.2
        assert layer.d_model == 128


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase2Integration:
    """Test Phase 2A + 2B → Phase 3 integration."""
    
    def test_with_phase2a_output(self):
        """Should work with Phase 2A adjacency matrix."""
        from panini_lm.phase2a_symbolic import build_adjacency_matrix
        from panini_lm.core.types import MorphToken
        
        # Create tokens and build adjacency
        tokens: list[MorphToken] = [
            {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta",
             "attributes": {"vibhakti": 1, "vacana": 1}},
            {"surface": "gacchati", "stem": "gam", "type": "tinanta",
             "attributes": {"vacana": 1}},
        ]
        
        adjacency = build_adjacency_matrix(tokens)
        
        # Create matching QKV
        batch, n_heads, d_head = 1, 4, 16
        seq_len = adjacency.matrix.shape[0]
        
        Q = torch.randn(batch, n_heads, seq_len, d_head)
        K = torch.randn(batch, n_heads, seq_len, d_head)
        V = torch.randn(batch, n_heads, seq_len, d_head)
        
        # Should work
        output = sparse_attention(Q, K, V, adjacency, training=False)
        assert output.shape == Q.shape
    
    def test_full_pipeline_shapes(self):
        """Test full Phase 1 → 2A → 2B → 3 shapes align."""
        from panini_lm.core.config import NeuralConfig
        from panini_lm.phase2a_symbolic import build_adjacency_matrix
        from panini_lm.phase2b_neural import process_neural
        
        tokens = [
            {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta",
             "attributes": {"vibhakti": 1, "vacana": 1}},
            {"surface": "gacchati", "stem": "gam", "type": "tinanta",
             "attributes": {"vacana": 1}},
        ]
        
        # Phase 2A: Build adjacency
        adjacency = build_adjacency_matrix(tokens)
        
        # Phase 2B: Get QKV
        config = NeuralConfig(d_model=64, num_heads=4)
        phase2b = process_neural(tokens, config=config)
        
        Q = phase2b["qkv"]["Q"]
        K = phase2b["qkv"]["K"]
        V = phase2b["qkv"]["V"]
        
        # Phase 3: Compute attention
        output = sparse_attention(Q, K, V, adjacency, training=False)
        
        # Output should match input shape
        assert output.shape == Q.shape
