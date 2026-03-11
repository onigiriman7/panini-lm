"""
Morphology-Aware Embedding Layer.

CRITICAL DESIGN DECISION: NO POSITIONAL ENCODING!
Sanskrit has free word order — position should not bias attention.
Grammar constraints from Phase 2A (Matrix M) handle word relationships.

Components:
- Token embedding: Standard learned embedding for vocab
- Type embedding: Embedding for morphological type (noun, verb, etc.)
- Feature embedding: Optional embedding for morphological features
"""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn

from panini_lm.core.types import MorphToken
from panini_lm.core.config import NeuralConfig


class MorphAwareEmbedding(nn.Module):
    """
    Embedding layer that combines token identity with morphological features.
    
    NO positional encoding by design — word order in Sanskrit is free,
    so position should not influence attention. Grammar rules (Phase 2A)
    provide necessary structural information.
    
    Components combined:
    - Token embedding: E_tok(token_id)
    - Type embedding: E_type(type_id)
    - Optional feature embedding per morphological attribute
    
    Final embedding: E = E_tok + E_type (+ E_features if enabled)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_types: int = 7,
        dropout: float = 0.1,
        combine_features: bool = True,
        num_vibhakti: int = 8,  # 8 cases
        num_vacana: int = 3,    # singular, dual, plural
        num_linga: int = 3,     # masculine, feminine, neuter
        num_lakara: int = 10,   # 10 verbal tenses/moods
    ):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of token vocabulary
            d_model: Model dimension
            num_types: Number of token types
            dropout: Dropout probability
            combine_features: Whether to add feature embeddings
            num_vibhakti: Number of cases (for nouns)
            num_vacana: Number of numbers (sg, du, pl)
            num_linga: Number of genders
            num_lakara: Number of verbal tenses/moods
        """
        super().__init__()
        
        self.d_model = d_model
        self.combine_features = combine_features
        
        # Core embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_embedding = nn.Embedding(num_types, d_model)
        
        # Feature embeddings (smaller dimension, then projected)
        if combine_features:
            feature_dim = d_model // 4
            self.vibhakti_embedding = nn.Embedding(num_vibhakti + 1, feature_dim)  # +1 for N/A
            self.vacana_embedding = nn.Embedding(num_vacana + 1, feature_dim)
            self.linga_embedding = nn.Embedding(num_linga + 1, feature_dim)
            self.lakara_embedding = nn.Embedding(num_lakara + 1, feature_dim)
            
            # Project feature concat back to d_model
            self.feature_projection = nn.Linear(feature_dim * 4, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize embedding weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        
        # Zero out padding embedding
        with torch.no_grad():
            self.token_embedding.weight[0].fill_(0)
        
        if self.combine_features:
            nn.init.normal_(self.vibhakti_embedding.weight, std=0.02)
            nn.init.normal_(self.vacana_embedding.weight, std=0.02)
            nn.init.normal_(self.linga_embedding.weight, std=0.02)
            nn.init.normal_(self.lakara_embedding.weight, std=0.02)
    
    @classmethod
    def from_config(cls, config: NeuralConfig, vocab_size: int) -> "MorphAwareEmbedding":
        """Create embedding from config."""
        return cls(
            vocab_size=vocab_size,
            d_model=config.d_model,
            dropout=config.embedding_dropout,
            combine_features=True,  # Always use morphological features
        )
    
    def forward(
        self,
        token_ids: torch.Tensor,
        type_ids: torch.Tensor,
        feature_ids: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute embeddings.
        
        Args:
            token_ids: Token IDs, shape (batch, seq_len)
            type_ids: Type IDs, shape (batch, seq_len)
            feature_ids: Optional dict with feature tensors
                - "vibhakti": (batch, seq_len)
                - "vacana": (batch, seq_len)
                - "linga": (batch, seq_len)
                - "lakara": (batch, seq_len)
        
        Returns:
            Embeddings, shape (batch, seq_len, d_model)
        """
        # Base embeddings
        tok_emb = self.token_embedding(token_ids)
        type_emb = self.type_embedding(type_ids)
        
        # Combine: token + type
        x = tok_emb + type_emb
        
        # Add feature embeddings if enabled
        if self.combine_features and feature_ids is not None:
            vibhakti = self.vibhakti_embedding(feature_ids.get("vibhakti", 
                torch.zeros_like(token_ids)))
            vacana = self.vacana_embedding(feature_ids.get("vacana",
                torch.zeros_like(token_ids)))
            linga = self.linga_embedding(feature_ids.get("linga",
                torch.zeros_like(token_ids)))
            lakara = self.lakara_embedding(feature_ids.get("lakara",
                torch.zeros_like(token_ids)))
            
            # Concat features and project
            features = torch.cat([vibhakti, vacana, linga, lakara], dim=-1)
            feature_emb = self.feature_projection(features)
            
            x = x + feature_emb
        
        # Normalize and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x
    
    def embed_tokens(
        self,
        tokens: List[MorphToken],
        tokenizer,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convenience method to embed MorphTokens directly.
        
        Args:
            tokens: List of MorphToken
            tokenizer: PaniniTokenizer instance
            device: Target device
        
        Returns:
            Embeddings, shape (1, seq_len, d_model)
        """
        device = device or next(self.parameters()).device
        
        # Encode via tokenizer
        token_ids, type_ids = tokenizer.encode(tokens, add_bos=False, add_eos=False)
        
        # Convert to tensors
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        type_tensor = torch.tensor([type_ids], dtype=torch.long, device=device)
        
        # Extract feature IDs
        feature_ids = None
        if self.combine_features:
            vibhakti_ids = []
            vacana_ids = []
            linga_ids = []
            lakara_ids = []
            
            for token in tokens:
                attrs = token.get("attributes", {})
                vibhakti_ids.append(attrs.get("vibhakti", 0))
                vacana_ids.append(attrs.get("vacana", 0))
                linga_ids.append(self._linga_to_id(attrs.get("linga", "")))
                lakara_ids.append(self._lakara_to_id(attrs.get("lakara", "")))
            
            feature_ids = {
                "vibhakti": torch.tensor([vibhakti_ids], dtype=torch.long, device=device),
                "vacana": torch.tensor([vacana_ids], dtype=torch.long, device=device),
                "linga": torch.tensor([linga_ids], dtype=torch.long, device=device),
                "lakara": torch.tensor([lakara_ids], dtype=torch.long, device=device),
            }
        
        return self(token_tensor, type_tensor, feature_ids)
    
    def _linga_to_id(self, linga: str) -> int:
        """Map grammatical gender to ID."""
        mapping = {"m": 1, "f": 2, "n": 3, "": 0}
        return mapping.get(linga, 0)
    
    def _lakara_to_id(self, lakara: str) -> int:
        """Map verb tense/mood to ID."""
        mapping = {
            "lat": 1, "lit": 2, "lut": 3, "lrt": 4, "lot": 5,
            "lan": 6, "lin": 7, "lun": 8, "lrn": 9, "": 0
        }
        return mapping.get(lakara, 0)
