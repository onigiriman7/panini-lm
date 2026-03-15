#!/usr/bin/env python3
"""
Training Data Builder — Build training dataset for Panini-LM.

Produces tensor-ready training data:
- token_ids: Integer IDs for embedding lookup
- type_ids: Token type encodings
- adjacency_edges: Sparse grammatical links for attention supervision
- target_ids: Labels for next-token prediction

Usage:
    from scripts.training_data_builder import TrainingDataBuilder
    
    builder = TrainingDataBuilder()
    builder.build_vocab("gita.txt")  # First pass: build vocabulary
    dataset = builder.process_file("gita.txt")  # Second pass: encode
    builder.save_dataset(dataset, "gita_training.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import sys

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gita_parser import GitaParser, Sentence, Chapter

# Panini-LM imports
from panini_lm.core.types import (
    MorphToken,
    Phase1Output,
    AdjacencyMatrix,
    GrammaticalLink,
)
from panini_lm.core.config import MorphologyConfig, SymbolicConfig
from panini_lm.phase1_morphology.orchestrator import ingest_morphology
from panini_lm.phase2a_symbolic.matrix_builder import build_adjacency_matrix
from panini_lm.phase2b_neural.tokenizer import PaniniTokenizer


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AdjacencyEdge(TypedDict):
    """A single edge in the sparse adjacency representation."""
    src: int
    """Source token index."""
    tgt: int
    """Target token index."""
    link_type: str
    """Grammatical relationship type."""


class TrainingSample(TypedDict):
    """A single training example ready for model training."""
    id: str
    """Unique identifier (e.g., 'ch02_s0047')."""
    
    chapter: int
    """Chapter number."""
    
    raw_text: str
    """Original Sanskrit text."""
    
    # === Tokenizer outputs (model inputs) ===
    token_ids: List[int]
    """Integer IDs for embedding lookup. Includes BOS/EOS."""
    
    type_ids: List[int]
    """Token type encodings (subanta=0, tinanta=1, etc.)."""
    
    # === Training targets ===
    target_ids: List[int]
    """Labels for next-token prediction (shifted token_ids)."""
    
    # === Attention supervision ===
    adjacency_edges: List[AdjacencyEdge]
    """Sparse grammatical edges (src, tgt, type). Used to supervise attention."""
    
    # === Metadata ===
    seq_len: int
    """Sequence length including special tokens."""
    
    num_edges: int
    """Number of valid adjacency edges."""
    
    sparsity: float
    """Fraction of valid edges (num_edges / seq_len²)."""
    
    # === Ground truth morphology (for analysis/debugging) ===
    tokens: List[MorphToken]
    """Full morphological analysis from Phase 1."""


class TrainingDataset(TypedDict):
    """Complete training dataset for Panini-LM."""
    metadata: Dict[str, Any]
    """Dataset metadata (source, version, creation time)."""
    
    vocab: Dict[str, int]
    """Token vocabulary (stem/surface → ID mapping)."""
    
    type_vocab: Dict[str, int]
    """Token type vocabulary."""
    
    chapters: List[Dict[str, Any]]
    """Chapter summaries."""
    
    samples: List[TrainingSample]
    """Training samples."""
    
    statistics: Dict[str, Any]
    """Dataset statistics."""


@dataclass
class TrainingDataBuilder:
    """
    Build training data for Panini-LM.
    
    Two-pass approach:
    1. build_vocab(): First pass to collect all stems/surfaces into vocabulary
    2. process_file(): Second pass to encode sentences with final vocabulary
    
    The output format is optimized for PyTorch DataLoader consumption.
    """
    
    config_morph: MorphologyConfig = None
    config_symbolic: SymbolicConfig = None
    tokenizer: PaniniTokenizer = field(default_factory=PaniniTokenizer)
    
    def __post_init__(self):
        if self.config_morph is None:
            self.config_morph = MorphologyConfig()
        if self.config_symbolic is None:
            self.config_symbolic = SymbolicConfig()
        if self.tokenizer is None:
            self.tokenizer = PaniniTokenizer()
    
    def build_vocab(
        self,
        filepath: str,
        min_count: int = 1,
    ) -> None:
        """
        First pass: Build vocabulary from all tokens in file.
        
        Args:
            filepath: Path to gita.txt
            min_count: Minimum frequency to include token (default: 1)
        """
        logger.info(f"Building vocabulary from {filepath}...")
        
        parser = GitaParser()
        chapters = parser.parse_file(filepath)
        
        stem_counts: Dict[str, int] = {}
        surface_counts: Dict[str, int] = {}
        
        for chapter in chapters:
            for sentence in chapter["sentences"]:
                try:
                    phase1 = ingest_morphology(sentence["text"], config=self.config_morph)
                    for token in phase1["tokens"]:
                        stem = token.get("stem", "")
                        surface = token.get("surface", "")
                        if stem:
                            stem_counts[stem] = stem_counts.get(stem, 0) + 1
                        if surface:
                            surface_counts[surface] = surface_counts.get(surface, 0) + 1
                except Exception:
                    continue
        
        # Add stems first (primary vocabulary)
        for stem, count in sorted(stem_counts.items(), key=lambda x: -x[1]):
            if count >= min_count:
                self.tokenizer.add_token(stem)
        
        # Add unseen surfaces as fallback
        for surface, count in sorted(surface_counts.items(), key=lambda x: -x[1]):
            if count >= min_count and surface not in self.tokenizer.vocab:
                self.tokenizer.add_token(surface)
        
        logger.info(f"Vocabulary built: {self.tokenizer.vocab_size} tokens")
    
    def process_sentence(
        self,
        sentence: Sentence,
        sentence_id: str,
    ) -> Optional[TrainingSample]:
        """
        Process a single sentence into a training sample.
        
        Args:
            sentence: Input sentence
            sentence_id: Unique identifier
            
        Returns:
            TrainingSample or None if processing failed
        """
        text = sentence["text"]
        
        try:
            # === Phase 1: Morphological Analysis ===
            phase1_output = ingest_morphology(text, config=self.config_morph)
            tokens = phase1_output["tokens"]
            
            if not tokens:
                logger.warning(f"No tokens extracted for: {text[:50]}...")
                return None
            
            # === Encode with tokenizer ===
            token_ids, type_ids = self.tokenizer.encode(
                tokens, add_bos=True, add_eos=True
            )
            
            # Target IDs: shifted token_ids for next-token prediction
            # Input:  [BOS, t1, t2, t3, EOS]
            # Target: [t1, t2, t3, EOS, PAD]
            target_ids = token_ids[1:] + [self.tokenizer.pad_id]
            
            # === Phase 2A: Symbolic Adjacency ===
            adjacency = build_adjacency_matrix(
                tokens, config=self.config_symbolic
            )
            
            # Convert links to sparse edge list (offset by 1 for BOS token)
            adjacency_edges: List[AdjacencyEdge] = []
            for link in adjacency.links:
                adjacency_edges.append(AdjacencyEdge(
                    src=link["source_idx"] + 1,  # +1 for BOS
                    tgt=link["target_idx"] + 1,
                    link_type=link["link_type"],
                ))
            
            seq_len = len(token_ids)
            num_edges = len(adjacency_edges)
            sparsity = num_edges / (seq_len * seq_len) if seq_len > 0 else 0.0
            
            return TrainingSample(
                id=sentence_id,
                chapter=sentence["chapter"],
                raw_text=text,
                token_ids=token_ids,
                type_ids=type_ids,
                target_ids=target_ids,
                adjacency_edges=adjacency_edges,
                seq_len=seq_len,
                num_edges=num_edges,
                sparsity=sparsity,
                tokens=tokens,
            )
            
        except Exception as e:
            logger.error(f"Failed to process: {text[:50]}... Error: {e}")
            return None
    
    def process_chapter(
        self,
        chapter: Chapter,
        max_sentences: Optional[int] = None,
    ) -> List[TrainingSample]:
        """Process all sentences in a chapter."""
        results = []
        sentences = chapter["sentences"]
        
        if max_sentences:
            sentences = sentences[:max_sentences]
        
        for i, sentence in enumerate(sentences):
            sentence_id = f"ch{chapter['number']:02d}_s{i+1:04d}"
            result = self.process_sentence(sentence, sentence_id)
            if result:
                results.append(result)
        
        return results
    
    def process_file(
        self,
        filepath: str,
        max_sentences: Optional[int] = None,
        chapters: Optional[List[int]] = None,
        build_vocab_first: bool = True,
    ) -> TrainingDataset:
        """
        Process the entire file into training data.
        
        Args:
            filepath: Path to gita.txt
            max_sentences: Maximum sentences per chapter (None = all)
            chapters: Specific chapters to process (None = all)
            build_vocab_first: Whether to build vocab first (default: True)
            
        Returns:
            TrainingDataset ready for model training
        """
        if build_vocab_first:
            self.build_vocab(filepath)
        
        parser = GitaParser()
        parsed_chapters = parser.parse_file(filepath)
        
        all_samples: List[TrainingSample] = []
        chapter_summaries: List[Dict[str, Any]] = []
        
        for chapter in parsed_chapters:
            ch_num = chapter["number"]
            
            if chapters and ch_num not in chapters:
                continue
            
            logger.info(f"Processing Chapter {ch_num}: {chapter['name']}")
            
            processed = self.process_chapter(chapter, max_sentences)
            all_samples.extend(processed)
            
            chapter_summaries.append({
                "number": ch_num,
                "name": chapter["name"],
                "sanskrit_name": chapter["sanskrit_name"],
                "sentences_total": len(chapter["sentences"]),
                "sentences_processed": len(processed),
            })
            
            logger.info(f"  → Processed {len(processed)} samples")
        
        # Compute statistics
        stats = self._compute_statistics(all_samples)
        
        return TrainingDataset(
            metadata={
                "source": filepath,
                "created_at": datetime.now().isoformat(),
                "panini_lm_version": "0.1.0",
                "vocab_size": self.tokenizer.vocab_size,
                "num_types": self.tokenizer.num_types,
            },
            vocab=self.tokenizer.vocab,
            type_vocab=self.tokenizer.type_vocab,
            chapters=chapter_summaries,
            samples=all_samples,
            statistics=stats,
        )
    
    def _compute_statistics(self, samples: List[TrainingSample]) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not samples:
            return {}
        
        seq_lens = [s["seq_len"] for s in samples]
        num_edges = [s["num_edges"] for s in samples]
        sparsities = [s["sparsity"] for s in samples]
        
        # Token frequency analysis
        token_counts: Dict[int, int] = {}
        for s in samples:
            for tid in s["token_ids"]:
                token_counts[tid] = token_counts.get(tid, 0) + 1
        
        unk_count = token_counts.get(self.tokenizer.unk_id, 0)
        total_tokens = sum(token_counts.values())
        unk_rate = unk_count / total_tokens if total_tokens > 0 else 0.0
        
        return {
            "total_samples": len(samples),
            "total_tokens": total_tokens,
            "unk_rate": unk_rate,
            "seq_len": {
                "min": min(seq_lens),
                "max": max(seq_lens),
                "mean": sum(seq_lens) / len(seq_lens),
            },
            "num_edges": {
                "min": min(num_edges),
                "max": max(num_edges),
                "mean": sum(num_edges) / len(num_edges),
            },
            "sparsity": {
                "min": min(sparsities),
                "max": max(sparsities),
                "mean": sum(sparsities) / len(sparsities),
            },
        }
    
    def save_dataset(self, dataset: TrainingDataset, filepath: str) -> None:
        """Save dataset to JSON file."""
        path = Path(filepath)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dict(dataset), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved dataset to {path} ({len(dataset['samples'])} samples)")
    
    @staticmethod
    def load_dataset(filepath: str) -> TrainingDataset:
        """Load dataset from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Panini-LM training dataset")
    parser.add_argument("input", nargs="?", default="gita.txt", help="Input file")
    parser.add_argument("-o", "--output", default="tests/data/gita_training.json", help="Output file")
    parser.add_argument("-n", "--max-sentences", type=int, help="Max sentences per chapter")
    parser.add_argument("-c", "--chapters", type=int, nargs="+", help="Specific chapters")
    parser.add_argument("--min-count", type=int, default=1, help="Min token count for vocab")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    builder = TrainingDataBuilder()
    dataset = builder.process_file(
        args.input,
        max_sentences=args.max_sentences,
        chapters=args.chapters,
    )
    
    builder.save_dataset(dataset, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Dataset Statistics")
    print("=" * 60)
    print(f"Vocabulary size: {dataset['metadata']['vocab_size']}")
    print(f"Token types: {dataset['metadata']['num_types']}")
    print()
    for key, val in dataset["statistics"].items():
        if isinstance(val, dict):
            print(f"{key}:")
            for k2, v2 in val.items():
                print(f"  {k2}: {v2:.4f}" if isinstance(v2, float) else f"  {k2}: {v2}")
        else:
            if isinstance(val, float):
                print(f"{key}: {val:.4f}")
            else:
                print(f"{key}: {val}")


if __name__ == "__main__":
    main()
