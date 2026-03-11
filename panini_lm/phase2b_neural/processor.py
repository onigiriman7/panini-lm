"""
Phase 2B Processor — Main entry point for neural processing.

Orchestrates tokenization, embedding, and QKV projection.
Produces Phase2BOutput that feeds into Phase 3 (Sparse Attention).

Usage:
    from panini_lm.phase2b_neural import process_neural
    
    result = process_neural(
        tokens=phase1_output["tokens"],
        config=neural_config,
    )
"""

from __future__ import annotations

from typing import List, Optional
from dataclasses import dataclass
import torch

from panini_lm.core.types import MorphToken, QKVTensors, Phase2BOutput
from panini_lm.core.config import NeuralConfig
from panini_lm.core.exceptions import NeuralEngineError

from panini_lm.phase2b_neural.tokenizer import PaniniTokenizer
from panini_lm.phase2b_neural.embedding import MorphAwareEmbedding
from panini_lm.phase2b_neural.projection import QKVProjection


@dataclass
class NeuralPipeline:
    """
    Container for Phase 2B neural components.
    
    Holds:
    - Tokenizer: Vocabulary management
    - Embedding: Token + type + feature embeddings
    - Projection: Q, K, V linear projections
    """
    
    tokenizer: PaniniTokenizer
    embedding: MorphAwareEmbedding
    projection: QKVProjection
    config: NeuralConfig
    device: torch.device
    
    @classmethod
    def from_config(
        cls,
        config: NeuralConfig,
        tokenizer: Optional[PaniniTokenizer] = None,
        device: Optional[torch.device] = None,
    ) -> "NeuralPipeline":
        """
        Create pipeline from config.
        
        Args:
            config: Neural configuration
            tokenizer: Optional pre-built tokenizer
            device: Target device
            
        Returns:
            Initialized NeuralPipeline
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create or use provided tokenizer
        tokenizer = tokenizer or PaniniTokenizer()
        vocab_size = max(tokenizer.vocab_size, 1000)  # Minimum vocab size
        
        # Create embedding and projection
        embedding = MorphAwareEmbedding.from_config(config, vocab_size)
        projection = QKVProjection.from_config(config)
        
        # Move to device
        embedding = embedding.to(device)
        projection = projection.to(device)
        
        return cls(
            tokenizer=tokenizer,
            embedding=embedding,
            projection=projection,
            config=config,
            device=device,
        )
    
    def process(
        self,
        tokens: List[MorphToken],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> Phase2BOutput:
        """
        Process tokens through neural pipeline.
        
        Args:
            tokens: List of MorphToken from Phase 1
            add_bos: Add begin-of-sequence token
            add_eos: Add end-of-sequence token
        
        Returns:
            Phase2BOutput with embeddings and QKV tensors
        """
        if not tokens:
            raise NeuralEngineError("Empty token list")
        
        # Encode tokens
        token_ids, type_ids = self.tokenizer.encode(
            tokens, 
            add_bos=add_bos, 
            add_eos=add_eos
        )
        
        # Convert to tensors
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        type_tensor = torch.tensor([type_ids], dtype=torch.long, device=self.device)
        
        # Extract feature IDs
        feature_ids = self._extract_features(tokens, add_bos, add_eos)
        
        # Get embeddings
        embeddings = self.embedding(token_tensor, type_tensor, feature_ids)
        
        # Get QKV projections
        qkv = self.projection(embeddings, return_dict=True)
        
        return {
            "embeddings": embeddings,
            "qkv": qkv,
            "token_ids": token_ids,
            "type_ids": type_ids,
            "seq_len": len(token_ids),
        }
    
    def _extract_features(
        self,
        tokens: List[MorphToken],
        add_bos: bool,
        add_eos: bool,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Extract morphological feature tensors."""
        # Always extract features for Panini LM
        vibhakti_ids = []
        vacana_ids = []
        linga_ids = []
        lakara_ids = []
        
        # Add padding for BOS if needed
        if add_bos:
            vibhakti_ids.append(0)
            vacana_ids.append(0)
            linga_ids.append(0)
            lakara_ids.append(0)
        
        for token in tokens:
            attrs = token.get("attributes", {})
            vibhakti_ids.append(attrs.get("vibhakti", 0))
            vacana_ids.append(attrs.get("vacana", 0))
            linga_ids.append(self._linga_to_id(attrs.get("linga", "")))
            lakara_ids.append(self._lakara_to_id(attrs.get("lakara", "")))
        
        # Add padding for EOS if needed
        if add_eos:
            vibhakti_ids.append(0)
            vacana_ids.append(0)
            linga_ids.append(0)
            lakara_ids.append(0)
        
        return {
            "vibhakti": torch.tensor([vibhakti_ids], dtype=torch.long, device=self.device),
            "vacana": torch.tensor([vacana_ids], dtype=torch.long, device=self.device),
            "linga": torch.tensor([linga_ids], dtype=torch.long, device=self.device),
            "lakara": torch.tensor([lakara_ids], dtype=torch.long, device=self.device),
        }
    
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


# Module-level convenience function
def process_neural(
    tokens: List[MorphToken],
    config: Optional[NeuralConfig] = None,
    tokenizer: Optional[PaniniTokenizer] = None,
    device: Optional[torch.device] = None,
) -> Phase2BOutput:
    """
    Process tokens through Phase 2B neural pipeline.
    
    This is the main entry point for Phase 2B.
    
    Args:
        tokens: List of MorphToken from Phase 1
        config: Neural configuration (uses defaults if None)
        tokenizer: Optional tokenizer (creates new if None)
        device: Target device
    
    Returns:
        Phase2BOutput containing embeddings and QKV tensors
    
    Example:
        >>> from panini_lm.phase2b_neural import process_neural
        >>> from panini_lm.core.config import NeuralConfig
        >>> 
        >>> tokens = [
        ...     {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta",
        ...      "attributes": {"vibhakti": 1, "vacana": 1}},
        ... ]
        >>> 
        >>> result = process_neural(tokens, config=NeuralConfig())
        >>> print(result["embeddings"].shape)
        torch.Size([1, 1, 256])
    """
    config = config or NeuralConfig()
    
    pipeline = NeuralPipeline.from_config(
        config=config,
        tokenizer=tokenizer,
        device=device,
    )
    
    return pipeline.process(tokens)
