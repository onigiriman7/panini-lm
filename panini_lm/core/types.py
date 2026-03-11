"""
Core type definitions for Panini-LM.

All inter-phase data contracts are defined here using TypedDict and dataclasses.
These serve as the single source of truth for data shapes across the pipeline.

Usage:
    from panini_lm.core.types import MorphToken, Phase1Output, AdjacencyMatrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, Literal, Optional, List, Dict, Any

import torch


# =============================================================================
# Phase 1: Morphological Ingestion Types
# =============================================================================

class MorphAttributes(TypedDict, total=False):
    """
    Grammatical attributes of a morphological token.
    
    All fields are optional as different token types have different attributes.
    See docs/GLOSSARY.md for detailed explanations of each attribute.
    """
    # === Nominal attributes (subanta) ===
    vibhakti: int
    """Case ending (1-8): 1=nominative, 2=accusative, etc."""
    
    linga: Literal["m", "f", "n"]
    """Gender: m=masculine, f=feminine, n=neuter."""
    
    # === Verbal attributes (tiṅanta) ===
    lakara: str
    """Tense/mood marker: laṭ (present), liṭ (perfect), etc."""
    
    # === Common attributes ===
    vacana: Literal[1, 2, 3]
    """Number: 1=singular (eka), 2=dual (dvi), 3=plural (bahu)."""
    
    purusa: Literal[1, 2, 3]
    """Person: 1=prathama (3rd), 2=madhyama (2nd), 3=uttama (1st)."""
    
    # === Semantic role (may be set in Phase 2A) ===
    karaka: Literal["karta", "karma", "karana", "sampradana", "apadana", "adhikarana"]
    """Semantic case role."""


TokenType = Literal["subanta", "tinanta", "avyaya", "krdanta", "unknown"]


class MorphToken(TypedDict):
    """
    A single morphologically analyzed token.
    
    This is the atomic unit output by Phase 1 and consumed by Phases 2A, 2B.
    
    Example:
        {
            "surface": "rāmaḥ",
            "stem": "rāma",
            "type": "subanta",
            "attributes": {"vibhakti": 1, "vacana": 1, "linga": "m"}
        }
    """
    surface: str
    """Surface form after sandhi resolution."""
    
    stem: str
    """Base form (prakṛti/dhātu)."""
    
    type: TokenType
    """Token category: subanta, tinanta, avyaya, krdanta, or unknown."""
    
    attributes: MorphAttributes
    """Grammatical attributes (varies by type)."""


class Phase1Output(TypedDict):
    """
    Complete output of Phase 1 Morphological Ingestion.
    
    Contains the analyzed tokens plus metadata for debugging and validation.
    """
    tokens: List[MorphToken]
    """Ordered list of analyzed tokens."""
    
    raw_input: str
    """Original input string (for debugging)."""
    
    sandhi_splits: List[str]
    """Intermediate sandhi-resolved forms."""


# =============================================================================
# Phase 2A: Symbolic Engine Types  
# =============================================================================

class AdjacencyMeta(TypedDict):
    """Metadata about the adjacency matrix."""
    seq_len: int
    """Sequence length N."""
    
    num_valid_edges: int
    """Number of valid (non -inf) edges."""
    
    sparsity_ratio: float
    """Fraction of edges that are valid: num_valid_edges / N²."""
    
    avg_connections_per_token: float
    """Average k value (target: 2-3)."""


class GrammaticalLink(TypedDict):
    """A single grammatical relationship between tokens."""
    source_idx: int
    """Index of the head/governor token."""
    
    target_idx: int
    """Index of the dependent token."""
    
    link_type: str
    """Relationship type (e.g., 'subject-verb', 'object-verb')."""
    
    rule_applied: str
    """Name of the grammar rule that licensed this link."""


@dataclass
class AdjacencyMatrix:
    """
    Sparse grammatical adjacency matrix M from Phase 2A.
    
    M[i,j] = 0.0 means token i can attend to token j.
    M[i,j] = -inf means token i cannot attend to token j.
    
    This matrix is added to attention scores before softmax.
    """
    matrix: torch.Tensor
    """Shape: (seq_len, seq_len). Values: 0.0 or -inf."""
    
    meta: AdjacencyMeta
    """Statistics about the matrix."""
    
    links: List[GrammaticalLink] = field(default_factory=list)
    """Explicit list of valid grammatical links (for debugging)."""
    
    def to(self, device: torch.device) -> "AdjacencyMatrix":
        """Move matrix to specified device."""
        return AdjacencyMatrix(
            matrix=self.matrix.to(device),
            meta=self.meta,
            links=self.links,
        )
    
    @property
    def sparsity(self) -> float:
        """Fraction of valid edges."""
        return self.meta["sparsity_ratio"]
    
    @property
    def avg_k(self) -> float:
        """Average connections per token."""
        return self.meta["avg_connections_per_token"]


class Phase2AOutput(TypedDict):
    """Output of Phase 2A Symbolic Engine."""
    adjacency_matrix: AdjacencyMatrix
    """The grammatical adjacency matrix M."""
    
    links: List[GrammaticalLink]
    """Explicit grammatical relationships."""


# =============================================================================
# Phase 2B: Neural Engine Types
# =============================================================================

class QKVTensors(TypedDict):
    """Query, Key, Value tensors for attention."""
    Q: torch.Tensor
    """Query tensor. Shape: (batch, heads, seq, head_dim)."""
    
    K: torch.Tensor
    """Key tensor. Shape: (batch, heads, seq, head_dim)."""
    
    V: torch.Tensor
    """Value tensor. Shape: (batch, heads, seq, head_dim)."""


class Phase2BOutput(TypedDict):
    """Output of Phase 2B Neural Engine."""
    embeddings: torch.Tensor
    """Raw token embeddings. Shape: (batch, seq, d_model)."""
    
    qkv: QKVTensors
    """Projected Q, K, V tensors."""


# =============================================================================
# Phase 3-4: Attention and FFN Types
# =============================================================================

class AttentionOutput(TypedDict):
    """Output of Phase 3 Sparse Attention."""
    hidden_states: torch.Tensor
    """Attention output. Shape: (batch, seq, d_model)."""
    
    attention_weights: Optional[torch.Tensor]
    """Attention weights (optional, for debugging). Shape: (batch, heads, seq, seq)."""


# =============================================================================
# Phase 5: Constrained Decoding Types
# =============================================================================

@dataclass
class MorphologicalState:
    """
    Tracks grammatical context during autoregressive generation.
    
    Used by Phase 5 to determine which tokens are legal continuations.
    """
    # Active incomplete relationships
    open_relations: List[str] = field(default_factory=list)
    """Pending grammatical relations (e.g., 'karma-pending')."""
    
    # Agreement context
    vacana_ctx: Optional[int] = None
    """Number agreement context (1=sg, 2=dual, 3=pl)."""
    
    purusa_ctx: Optional[int] = None
    """Person agreement context."""
    
    # Expected patterns
    expected_vibhakti: Optional[List[int]] = None
    """Legal case endings for next noun."""
    
    # Compound state
    in_compound: bool = False
    """Whether currently inside a compound word."""
    
    compound_members: List[str] = field(default_factory=list)
    """Stems accumulated in current compound."""
    
    def copy(self) -> "MorphologicalState":
        """Create a deep copy of this state."""
        return MorphologicalState(
            open_relations=list(self.open_relations),
            vacana_ctx=self.vacana_ctx,
            purusa_ctx=self.purusa_ctx,
            expected_vibhakti=list(self.expected_vibhakti) if self.expected_vibhakti else None,
            in_compound=self.in_compound,
            compound_members=list(self.compound_members),
        )


@dataclass
class GrammarMask:
    """
    Vocabulary mask based on grammatical constraints.
    
    mask[i] = 0.0 means token i is legal.
    mask[i] = -inf means token i is illegal.
    """
    mask: torch.Tensor
    """Shape: (vocab_size,). Values: 0.0 or -inf."""
    
    legal_count: int
    """Number of legal tokens."""
    
    @property
    def illegal_count(self) -> int:
        """Number of illegal tokens."""
        return self.mask.shape[0] - self.legal_count


@dataclass
class GenerationOutput:
    """Output of constrained generation."""
    tokens: List[MorphToken]
    """Generated tokens (guaranteed grammatical)."""
    
    text: str
    """Detokenized output text."""
    
    stats: Dict[str, Any] = field(default_factory=dict)
    """Generation statistics (rejected count, etc.)."""


# =============================================================================
# Utility Types
# =============================================================================

@dataclass
class TokenBatch:
    """A batch of tokenized sequences for model input."""
    token_ids: torch.Tensor
    """Token IDs. Shape: (batch, seq)."""
    
    attention_mask: torch.Tensor
    """Padding mask. Shape: (batch, seq). 1=valid, 0=padding."""
    
    morph_tokens: Optional[List[List[MorphToken]]] = None
    """Original MorphTokens (for Phase 2A)."""
    
    def to(self, device: torch.device) -> "TokenBatch":
        """Move batch to specified device."""
        return TokenBatch(
            token_ids=self.token_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            morph_tokens=self.morph_tokens,
        )
