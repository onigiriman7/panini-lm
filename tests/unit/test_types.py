"""Tests for core types."""

import pytest
import torch

from panini_lm.core.types import (
    MorphToken,
    MorphAttributes,
    Phase1Output,
    AdjacencyMatrix,
    AdjacencyMeta,
    QKVTensors,
    MorphologicalState,
    GrammarMask,
    TokenBatch,
)


class TestMorphToken:
    """Tests for MorphToken type."""
    
    def test_create_subanta_token(self):
        """Create a nominal (subanta) token."""
        token: MorphToken = {
            "surface": "rāmaḥ",
            "stem": "rāma",
            "type": "subanta",
            "attributes": {
                "vibhakti": 1,
                "vacana": 1,
                "linga": "m",
            }
        }
        assert token["surface"] == "rāmaḥ"
        assert token["stem"] == "rāma"
        assert token["type"] == "subanta"
        assert token["attributes"]["vibhakti"] == 1
    
    def test_create_tinanta_token(self):
        """Create a verbal (tinanta) token."""
        token: MorphToken = {
            "surface": "gacchati",
            "stem": "gam",
            "type": "tinanta",
            "attributes": {
                "purusa": 1,
                "vacana": 1,
                "lakara": "lat",
            }
        }
        assert token["type"] == "tinanta"
        assert token["attributes"]["lakara"] == "lat"
    
    def test_create_avyaya_token(self):
        """Create an indeclinable (avyaya) token."""
        token: MorphToken = {
            "surface": "ca",
            "stem": "ca",
            "type": "avyaya",
            "attributes": {}
        }
        assert token["type"] == "avyaya"
        assert len(token["attributes"]) == 0


class TestPhase1Output:
    """Tests for Phase1Output type."""
    
    def test_create_phase1_output(self):
        """Create complete Phase 1 output."""
        output: Phase1Output = {
            "raw_input": "rāmaḥ gacchati",
            "sandhi_splits": ["rāmaḥ", "gacchati"],
            "tokens": [
                {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta", 
                 "attributes": {"vibhakti": 1}},
                {"surface": "gacchati", "stem": "gam", "type": "tinanta",
                 "attributes": {"lakara": "lat"}},
            ]
        }
        assert len(output["tokens"]) == 2
        assert output["raw_input"] == "rāmaḥ gacchati"


class TestAdjacencyMatrix:
    """Tests for AdjacencyMatrix dataclass."""
    
    def test_create_matrix(self):
        """Create adjacency matrix with metadata."""
        N = 4
        matrix = torch.zeros(N, N)
        matrix[0, 2] = float('-inf')  # Block position 0→2
        
        meta: AdjacencyMeta = {
            "seq_len": N,
            "num_valid_edges": 15,
            "sparsity_ratio": 15 / 16,
            "avg_connections_per_token": 3.75,
        }
        
        adj = AdjacencyMatrix(matrix=matrix, meta=meta)
        
        assert adj.matrix.shape == (4, 4)
        assert adj.sparsity == 15 / 16
        assert adj.avg_k == 3.75
    
    def test_move_to_device(self):
        """Test moving matrix to different device."""
        matrix = torch.zeros(2, 2)
        meta: AdjacencyMeta = {
            "seq_len": 2,
            "num_valid_edges": 4,
            "sparsity_ratio": 1.0,
            "avg_connections_per_token": 2.0,
        }
        adj = AdjacencyMatrix(matrix=matrix, meta=meta)
        
        # Move to CPU (should work on any system)
        adj_cpu = adj.to(torch.device("cpu"))
        assert adj_cpu.matrix.device.type == "cpu"


class TestMorphologicalState:
    """Tests for MorphologicalState dataclass."""
    
    def test_default_state(self):
        """Create state with defaults."""
        state = MorphologicalState()
        assert state.open_relations == []
        assert state.vacana_ctx is None
        assert state.in_compound is False
    
    def test_state_copy(self):
        """Deep copy should be independent."""
        state = MorphologicalState(
            open_relations=["karma-pending"],
            vacana_ctx=1,
        )
        
        copy = state.copy()
        copy.open_relations.append("karta-pending")
        copy.vacana_ctx = 3
        
        # Original should be unchanged
        assert len(state.open_relations) == 1
        assert state.vacana_ctx == 1


class TestGrammarMask:
    """Tests for GrammarMask dataclass."""
    
    def test_create_mask(self):
        """Create grammar mask."""
        vocab_size = 1000
        mask = torch.zeros(vocab_size)
        mask[100:200] = float('-inf')  # Block tokens 100-199
        
        grammar_mask = GrammarMask(mask=mask, legal_count=900)
        
        assert grammar_mask.legal_count == 900
        assert grammar_mask.illegal_count == 100


class TestTokenBatch:
    """Tests for TokenBatch dataclass."""
    
    def test_create_batch(self):
        """Create token batch."""
        batch_size, seq_len = 4, 32
        
        batch = TokenBatch(
            token_ids=torch.randint(0, 1000, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len),
        )
        
        assert batch.token_ids.shape == (4, 32)
        assert batch.attention_mask.shape == (4, 32)
    
    def test_batch_to_device(self):
        """Test moving batch to device."""
        batch = TokenBatch(
            token_ids=torch.zeros(2, 10, dtype=torch.long),
            attention_mask=torch.ones(2, 10),
        )
        
        batch_cpu = batch.to(torch.device("cpu"))
        assert batch_cpu.token_ids.device.type == "cpu"
