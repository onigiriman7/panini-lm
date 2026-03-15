#!/usr/bin/env python3
"""
Visualizer — Visualize Phase 1 and Phase 2 outputs.

Provides ASCII and rich formatting for debugging and understanding
the transformation of Sanskrit text through the Panini-LM pipeline.

Usage:
    from scripts.visualizer import PipelineVisualizer
    
    viz = PipelineVisualizer()
    viz.visualize_sentence("कर्मणि एव ते अधिकारः")
    viz.visualize_from_dataset("tests/data/gita_training.json", sentence_id="ch02_s0042")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from panini_lm.core.types import MorphToken, GrammaticalLink
from panini_lm.phase1_morphology.orchestrator import ingest_morphology
from panini_lm.phase2a_symbolic.matrix_builder import build_adjacency_matrix, visualize_matrix


class PipelineVisualizer:
    """
    Visualizer for Panini-LM pipeline outputs.
    
    Provides multiple visualization modes:
    - Phase 1: Token table with morphological analysis
    - Phase 2A: Adjacency matrix with grammatical links
    - Combined: Full pipeline view
    """
    
    # Box drawing characters
    BOX_H = "─"
    BOX_V = "│"
    BOX_TL = "┌"
    BOX_TR = "┐"
    BOX_BL = "└"
    BOX_BR = "┘"
    BOX_T = "┬"
    BOX_B = "┴"
    BOX_L = "├"
    BOX_R = "┤"
    BOX_X = "┼"
    
    def _box(self, title: str, content: str, width: int = 80) -> str:
        """Create a box around content."""
        lines = content.split('\n')
        max_len = max(len(line) for line in lines) if lines else 0
        inner_width = max(max_len, len(title), width - 4)
        
        result = []
        result.append(f"{self.BOX_TL}{self.BOX_H * (inner_width + 2)}{self.BOX_TR}")
        result.append(f"{self.BOX_V} {title.center(inner_width)} {self.BOX_V}")
        result.append(f"{self.BOX_L}{self.BOX_H * (inner_width + 2)}{self.BOX_R}")
        
        for line in lines:
            result.append(f"{self.BOX_V} {line.ljust(inner_width)} {self.BOX_V}")
        
        result.append(f"{self.BOX_BL}{self.BOX_H * (inner_width + 2)}{self.BOX_BR}")
        
        return '\n'.join(result)
    
    def visualize_phase1(self, tokens: List[MorphToken], raw_text: str = "") -> str:
        """
        Visualize Phase 1 output: morphological tokens.
        
        Shows a table with:
        - Surface form
        - Stem
        - Token type
        - Key attributes
        """
        lines = []
        
        if raw_text:
            lines.append(f"Input: {raw_text}")
            lines.append("")
        
        # Header
        header = f"{'#':<3} {'Surface':<15} {'Stem':<12} {'Type':<10} {'Attributes':<30}"
        lines.append(header)
        lines.append("─" * len(header))
        
        for i, token in enumerate(tokens):
            surface = token.get("surface", "")[:14]
            stem = token.get("stem", "")[:11]
            ttype = token.get("type", "unknown")[:9]
            
            # Format attributes
            attrs = token.get("attributes", {})
            attr_parts = []
            if "vibhakti" in attrs:
                attr_parts.append(f"vib={attrs['vibhakti']}")
            if "vacana" in attrs:
                attr_parts.append(f"vac={attrs['vacana']}")
            if "linga" in attrs:
                attr_parts.append(f"liṅ={attrs['linga']}")
            if "lakara" in attrs:
                attr_parts.append(f"lak={attrs['lakara']}")
            if "karaka" in attrs:
                attr_parts.append(f"kār={attrs['karaka']}")
            
            attr_str = ", ".join(attr_parts)[:29] if attr_parts else "(none)"
            
            lines.append(f"{i:<3} {surface:<15} {stem:<12} {ttype:<10} {attr_str:<30}")
        
        return self._box("PHASE 1: Morphological Tokens", '\n'.join(lines))
    
    def visualize_phase2a(
        self,
        tokens: List[MorphToken],
        links: List[GrammaticalLink],
        matrix_viz: str = "",
    ) -> str:
        """
        Visualize Phase 2A output: adjacency matrix and grammatical links.
        """
        lines = []
        
        # Grammatical links
        lines.append("Grammatical Links:")
        lines.append("─" * 50)
        
        if links:
            for link in links:
                src = tokens[link["source_idx"]]["surface"][:10] if link["source_idx"] < len(tokens) else "?"
                tgt = tokens[link["target_idx"]]["surface"][:10] if link["target_idx"] < len(tokens) else "?"
                lines.append(
                    f"  [{link['source_idx']}] {src} → [{link['target_idx']}] {tgt} "
                    f"({link['link_type']}, rule: {link['rule_applied']})"
                )
        else:
            lines.append("  (no links found)")
        
        lines.append("")
        
        # Matrix visualization
        if matrix_viz:
            lines.append("Adjacency Matrix (✓ = can attend, ─ = blocked):")
            lines.append("─" * 50)
            lines.append(matrix_viz)
        
        return self._box("PHASE 2A: Symbolic Engine", '\n'.join(lines))
    
    def visualize_phase2b(
        self,
        token_ids: List[int],
        type_ids: List[int],
        embedding_shape: List[int],
    ) -> str:
        """Visualize Phase 2B output: neural embeddings."""
        lines = []
        
        if token_ids:
            lines.append("Token IDs:")
            lines.append(f"  {token_ids}")
            lines.append("")
            lines.append("Type IDs:")
            lines.append(f"  {type_ids}")
            lines.append("")
            lines.append("Embedding Shape:")
            lines.append(f"  {embedding_shape} (batch, seq, d_model)")
        else:
            lines.append("(Phase 2B not processed - run with --phase2b flag)")
        
        return self._box("PHASE 2B: Neural Engine", '\n'.join(lines))
    
    def visualize_sentence(
        self,
        text: str,
        include_phase2b: bool = False,
    ) -> str:
        """
        Process and visualize a single sentence through the entire pipeline.
        
        Args:
            text: Sanskrit text to process
            include_phase2b: Whether to include Phase 2B visualization
            
        Returns:
            Complete visualization string
        """
        sections = []
        
        # Phase 1
        phase1_output = ingest_morphology(text)
        tokens = phase1_output["tokens"]
        
        sections.append(self.visualize_phase1(tokens, text))
        
        # Phase 2A
        if tokens:
            adjacency = build_adjacency_matrix(tokens)
            matrix_viz = visualize_matrix(adjacency, tokens)
            sections.append(self.visualize_phase2a(tokens, adjacency.links, matrix_viz))
        
        # Phase 2B (optional)
        if include_phase2b and tokens:
            try:
                from panini_lm.phase2b_neural.processor import NeuralPipeline
                from panini_lm.core.config import NeuralConfig
                
                config = NeuralConfig()
                pipeline = NeuralPipeline.from_config(config)
                phase2b_output = pipeline.process(tokens)
                
                sections.append(self.visualize_phase2b(
                    phase2b_output.get("token_ids", []),
                    phase2b_output.get("type_ids", []),
                    list(phase2b_output["embeddings"].shape),
                ))
            except Exception as e:
                sections.append(self._box("PHASE 2B: Neural Engine", f"Error: {e}"))
        
        return '\n\n'.join(sections)
    
    def visualize_from_dataset(
        self,
        dataset_path: str,
        sentence_id: Optional[str] = None,
        chapter: Optional[int] = None,
        index: Optional[int] = None,
    ) -> str:
        """
        Visualize a sentence from a saved dataset.
        
        Args:
            dataset_path: Path to training dataset JSON
            sentence_id: Specific sentence ID (e.g., "ch02_s0042")
            chapter: Chapter number to find first sentence
            index: Index in the sentences list
            
        Returns:
            Visualization string
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        sentences = dataset["sentences"]
        
        # Find the sentence
        sentence = None
        
        if sentence_id:
            for s in sentences:
                if s["id"] == sentence_id:
                    sentence = s
                    break
        elif chapter:
            for s in sentences:
                if s["chapter"] == chapter:
                    sentence = s
                    break
        elif index is not None and 0 <= index < len(sentences):
            sentence = sentences[index]
        else:
            # Default to first sentence
            sentence = sentences[0] if sentences else None
        
        if not sentence:
            return "Sentence not found"
        
        # Visualize
        sections = []
        
        sections.append(self._box(
            f"Sentence: {sentence['id']}",
            f"Chapter {sentence['chapter']}\n{sentence['raw_text']}"
        ))
        
        sections.append(self.visualize_phase1(sentence["tokens"], sentence["raw_text"]))
        
        sections.append(self.visualize_phase2a(
            sentence["tokens"],
            sentence["grammatical_links"],
            sentence["matrix_visualization"],
        ))
        
        if sentence.get("token_ids"):
            sections.append(self.visualize_phase2b(
                sentence["token_ids"],
                sentence["type_ids"],
                sentence["embedding_shape"],
            ))
        
        return '\n\n'.join(sections)
    
    def visualize_dataset_summary(self, dataset_path: str) -> str:
        """Show dataset summary."""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        lines = []
        
        # Metadata
        meta = dataset.get("metadata", {})
        lines.append(f"Source: {meta.get('source', 'unknown')}")
        lines.append(f"Created: {meta.get('created_at', 'unknown')}")
        lines.append("")
        
        # Chapters
        lines.append("Chapters:")
        lines.append("─" * 50)
        for ch in dataset.get("chapters", []):
            lines.append(
                f"  Ch {ch['number']:2d}: {ch['sentences_processed']:3d}/{ch['sentences_total']:3d} sentences"
            )
        lines.append("")
        
        # Statistics
        stats = dataset.get("statistics", {})
        lines.append("Statistics:")
        lines.append("─" * 50)
        lines.append(f"  Total sentences: {stats.get('total_sentences', 0)}")
        lines.append(f"  Total tokens: {stats.get('total_tokens', 0)}")
        
        tps = stats.get("tokens_per_sentence", {})
        if tps:
            lines.append(f"  Tokens/sentence: min={tps.get('min', 0)}, max={tps.get('max', 0)}, mean={tps.get('mean', 0):.1f}")
        
        sp = stats.get("sparsity", {})
        if sp:
            lines.append(f"  Sparsity: min={sp.get('min', 0):.2%}, max={sp.get('max', 0):.2%}, mean={sp.get('mean', 0):.2%}")
        
        ak = stats.get("avg_connections_per_token", {})
        if ak:
            lines.append(f"  Avg k: min={ak.get('min', 0):.1f}, max={ak.get('max', 0):.1f}, mean={ak.get('mean', 0):.1f}")
        
        return self._box("Dataset Summary", '\n'.join(lines))


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Panini-LM pipeline outputs")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Sentence command
    sent_parser = subparsers.add_parser("sentence", help="Visualize a sentence")
    sent_parser.add_argument("text", help="Sanskrit text to process")
    sent_parser.add_argument("--phase2b", action="store_true", help="Include Phase 2B")
    
    # Dataset command
    data_parser = subparsers.add_parser("dataset", help="Visualize from dataset")
    data_parser.add_argument("path", help="Dataset JSON path")
    data_parser.add_argument("-i", "--id", help="Sentence ID")
    data_parser.add_argument("-c", "--chapter", type=int, help="Chapter number")
    data_parser.add_argument("-n", "--index", type=int, help="Sentence index")
    data_parser.add_argument("--summary", action="store_true", help="Show summary only")
    
    args = parser.parse_args()
    
    viz = PipelineVisualizer()
    
    if args.command == "sentence":
        print(viz.visualize_sentence(args.text, include_phase2b=args.phase2b))
    
    elif args.command == "dataset":
        if args.summary:
            print(viz.visualize_dataset_summary(args.path))
        else:
            print(viz.visualize_from_dataset(
                args.path,
                sentence_id=args.id,
                chapter=args.chapter,
                index=args.index,
            ))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
