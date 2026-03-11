"""
Tests for Phase 2A: Symbolic Engine.

Test categories:
1. Rule tests — individual grammar rule behavior
2. Matrix builder tests — adjacency matrix construction
3. Sparsity tests — verify expected sparsity levels
4. Integration tests — Phase 1 → Phase 2A pipeline

Run with: pytest tests/unit/test_phase2a.py -v
"""

import pytest
import torch

from panini_lm.core.types import MorphToken, AdjacencyMatrix
from panini_lm.phase2a_symbolic import (
    build_adjacency_matrix,
    compute_adjacency_meta,
    GrammarRule,
    SubjectVerbRule,
    ObjectVerbRule,
    SelfAttentionRule,
    get_default_rules,
)
from panini_lm.phase2a_symbolic.rules import (
    VerbSubjectRule,
    VerbObjectRule,
    ParticleRule,
    AdjacentRule,
)
from panini_lm.phase2a_symbolic.matrix_builder import visualize_matrix


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def subject_token() -> MorphToken:
    """Nominative noun token (subject)."""
    return {
        "surface": "rāmaḥ",
        "stem": "rāma",
        "type": "subanta",
        "attributes": {"vibhakti": 1, "vacana": 1, "linga": "m"}
    }


@pytest.fixture
def object_token() -> MorphToken:
    """Accusative noun token (object)."""
    return {
        "surface": "gṛham",
        "stem": "gṛha",
        "type": "subanta",
        "attributes": {"vibhakti": 2, "vacana": 1, "linga": "n"}
    }


@pytest.fixture
def verb_token() -> MorphToken:
    """Verb token."""
    return {
        "surface": "gacchati",
        "stem": "gam",
        "type": "tinanta",
        "attributes": {"purusa": 1, "vacana": 1, "lakara": "lat"}
    }


@pytest.fixture
def particle_token() -> MorphToken:
    """Particle (indeclinable) token."""
    return {
        "surface": "ca",
        "stem": "ca",
        "type": "avyaya",
        "attributes": {}
    }


@pytest.fixture
def sample_tokens(subject_token, object_token, verb_token) -> list[MorphToken]:
    """Sample sentence: rāmaḥ gṛham gacchati (Rama goes home)."""
    return [subject_token, object_token, verb_token]


# =============================================================================
# Rule Unit Tests
# =============================================================================

class TestSelfAttentionRule:
    """Tests for self-attention rule."""
    
    def test_self_attention_allowed(self, subject_token):
        """Token should be able to attend to itself."""
        rule = SelfAttentionRule()
        assert rule.check(subject_token, subject_token, 0, 0) is True
    
    def test_different_positions_not_allowed(self, subject_token, verb_token):
        """Different positions should not match self-attention."""
        rule = SelfAttentionRule()
        assert rule.check(subject_token, verb_token, 0, 1) is False
    
    def test_highest_priority(self):
        """Self-attention should have highest priority (0)."""
        rule = SelfAttentionRule()
        assert rule.priority == 0


class TestSubjectVerbRule:
    """Tests for subject-verb agreement rule."""
    
    def test_nominative_can_attend_verb(self, subject_token, verb_token):
        """Nominative noun can attend to verb."""
        rule = SubjectVerbRule()
        assert rule.check(subject_token, verb_token, 0, 1) is True
    
    def test_accusative_cannot_attend_as_subject(self, object_token, verb_token):
        """Accusative noun cannot use subject-verb rule."""
        rule = SubjectVerbRule()
        assert rule.check(object_token, verb_token, 0, 1) is False
    
    def test_number_agreement_required(self, verb_token):
        """Subject and verb must agree in number."""
        rule = SubjectVerbRule()
        
        # Singular subject
        sg_subject: MorphToken = {
            "surface": "devaḥ",
            "stem": "deva",
            "type": "subanta",
            "attributes": {"vibhakti": 1, "vacana": 1}
        }
        
        # Plural verb
        pl_verb: MorphToken = {
            "surface": "gacchanti",
            "stem": "gam",
            "type": "tinanta",
            "attributes": {"vacana": 3}
        }
        
        # Should fail due to number mismatch
        assert rule.check(sg_subject, pl_verb, 0, 1) is False
    
    def test_number_agreement_passes(self, subject_token, verb_token):
        """Matching number should pass."""
        rule = SubjectVerbRule()
        # Both are vacana=1 (singular)
        assert rule.check(subject_token, verb_token, 0, 1) is True


class TestObjectVerbRule:
    """Tests for object-verb rule."""
    
    def test_accusative_can_attend_verb(self, object_token, verb_token):
        """Accusative noun can attend to verb."""
        rule = ObjectVerbRule()
        assert rule.check(object_token, verb_token, 0, 1) is True
    
    def test_nominative_cannot_use_object_rule(self, subject_token, verb_token):
        """Nominative cannot use object-verb rule."""
        rule = ObjectVerbRule()
        assert rule.check(subject_token, verb_token, 0, 1) is False


class TestParticleRule:
    """Tests for particle (avyaya) rule."""
    
    def test_particle_attends_adjacent(self, particle_token, verb_token):
        """Particle can attend to adjacent tokens."""
        rule = ParticleRule()
        # Adjacent (distance = 1)
        assert rule.check(particle_token, verb_token, 0, 1) is True
    
    def test_particle_not_distant(self, particle_token, verb_token):
        """Particle cannot attend to distant tokens."""
        rule = ParticleRule()
        # Distance = 3
        assert rule.check(particle_token, verb_token, 0, 3) is False


# =============================================================================
# Matrix Builder Tests
# =============================================================================

class TestBuildAdjacencyMatrix:
    """Tests for adjacency matrix construction."""
    
    def test_empty_input(self):
        """Empty token list should return empty matrix."""
        adj = build_adjacency_matrix([])
        
        assert adj.matrix.shape == (0, 0)
        assert adj.meta["seq_len"] == 0
    
    def test_single_token(self, subject_token):
        """Single token should have self-attention."""
        adj = build_adjacency_matrix([subject_token])
        
        assert adj.matrix.shape == (1, 1)
        assert adj.matrix[0, 0] == 0.0  # Self-attention
    
    def test_subject_verb_sentence(self, subject_token, verb_token):
        """Subject-verb sentence should have expected links."""
        tokens = [subject_token, verb_token]
        adj = build_adjacency_matrix(tokens)
        
        assert adj.matrix.shape == (2, 2)
        
        # Self-attention
        assert adj.matrix[0, 0] == 0.0
        assert adj.matrix[1, 1] == 0.0
        
        # Subject → Verb (kartā-kriyā)
        assert adj.matrix[0, 1] == 0.0
        
        # Verb → Subject (kriyā-kartā)
        assert adj.matrix[1, 0] == 0.0
    
    def test_three_word_sentence(self, sample_tokens):
        """Full S-O-V sentence."""
        adj = build_adjacency_matrix(sample_tokens)
        
        assert adj.matrix.shape == (3, 3)
        
        # Subject (0) → Verb (2)
        assert adj.matrix[0, 2] == 0.0
        
        # Object (1) → Verb (2)
        assert adj.matrix[1, 2] == 0.0
        
        # All self-attention
        for i in range(3):
            assert adj.matrix[i, i] == 0.0
    
    def test_links_recorded(self, sample_tokens):
        """Links should be recorded with rule names."""
        adj = build_adjacency_matrix(sample_tokens)
        
        # Should have multiple links
        assert len(adj.links) > 0
        
        # Check link structure
        for link in adj.links:
            assert "source_idx" in link
            assert "target_idx" in link
            assert "link_type" in link
            assert "rule_applied" in link
    
    def test_device_placement(self, sample_tokens):
        """Matrix should be on specified device."""
        adj = build_adjacency_matrix(sample_tokens, device=torch.device("cpu"))
        assert adj.matrix.device.type == "cpu"


class TestComputeAdjacencyMeta:
    """Tests for metadata computation."""
    
    def test_full_connectivity(self):
        """Full matrix (all 0.0) should have sparsity 1.0."""
        matrix = torch.zeros(4, 4)
        meta = compute_adjacency_meta(matrix)
        
        assert meta["sparsity_ratio"] == 1.0
        assert meta["num_valid_edges"] == 16
        assert meta["avg_connections_per_token"] == 4.0
    
    def test_diagonal_only(self):
        """Diagonal-only matrix (self-attention only)."""
        N = 4
        matrix = torch.full((N, N), float('-inf'))
        for i in range(N):
            matrix[i, i] = 0.0
        
        meta = compute_adjacency_meta(matrix)
        
        assert meta["num_valid_edges"] == N
        assert meta["avg_connections_per_token"] == 1.0
        assert meta["sparsity_ratio"] == N / (N * N)
    
    def test_empty_matrix(self):
        """Empty matrix should have zero everywhere."""
        matrix = torch.zeros(0, 0)
        meta = compute_adjacency_meta(matrix)
        
        assert meta["seq_len"] == 0
        assert meta["num_valid_edges"] == 0


class TestSparsity:
    """Tests for expected sparsity levels."""
    
    def test_typical_sparsity(self, sample_tokens):
        """Typical sentence should have sparse matrix (k ≈ 2-5)."""
        adj = build_adjacency_matrix(sample_tokens)
        
        # Average connections should be reasonable
        avg_k = adj.meta["avg_connections_per_token"]
        assert 1.0 <= avg_k <= 5.0, f"Unexpected avg_k: {avg_k}"
    
    def test_sparsity_increases_with_length(self):
        """Longer sequences should maintain reasonable sparsity."""
        # Create longer sequence with alternating nouns and verbs
        tokens = []
        for i in range(10):
            if i % 2 == 0:
                tokens.append({
                    "surface": f"noun{i}",
                    "stem": f"noun{i}",
                    "type": "subanta",
                    "attributes": {"vibhakti": 1, "vacana": 1}
                })
            else:
                tokens.append({
                    "surface": f"verb{i}",
                    "stem": f"verb{i}",
                    "type": "tinanta",
                    "attributes": {"vacana": 1}
                })
        
        adj = build_adjacency_matrix(tokens)
        
        # Should not have O(N²) connected
        # With N=10, full connectivity would be 100 edges
        # Sparse should be less than full (adjacent rule allows some extra)
        assert adj.meta["num_valid_edges"] < 80


class TestVisualization:
    """Tests for matrix visualization."""
    
    def test_visualize_matrix(self, sample_tokens):
        """Visualization should produce readable output."""
        adj = build_adjacency_matrix(sample_tokens)
        viz = visualize_matrix(adj, sample_tokens)
        
        # Should be non-empty string
        assert len(viz) > 0
        
        # Should contain token surfaces
        for token in sample_tokens:
            surface = token["surface"][:6]
            assert surface in viz
    
    def test_visualize_empty(self):
        """Empty matrix visualization."""
        adj = build_adjacency_matrix([])
        viz = visualize_matrix(adj, [])
        
        assert "empty" in viz.lower()


class TestDefaultRules:
    """Tests for default rule set."""
    
    def test_rules_sorted_by_priority(self):
        """Rules should be sorted by priority."""
        rules = get_default_rules()
        
        priorities = [r.priority for r in rules]
        assert priorities == sorted(priorities)
    
    def test_self_attention_first(self):
        """Self-attention should have highest priority."""
        rules = get_default_rules()
        
        assert rules[0].name == "sva-sambandha"
        assert isinstance(rules[0], SelfAttentionRule)
    
    def test_all_rules_have_names(self):
        """All rules should have names."""
        rules = get_default_rules()
        
        for rule in rules:
            assert len(rule.name) > 0
