#!/usr/bin/env python3
"""
Gita Training Data Builder — Convert gita.txt to Panini-LM training format.

Produces factorized tensor training data following the Panini-LM specification:
- root_ids: Root/stem IDs (~4000 vocabulary)
- type_ids: Token type encodings (subanta=0, tiṅanta=1, avyaya=2, etc.)
- vibhakti_ids: Case endings (0=none, 1-7=cases, 8=vocative)
- vacana_ids: Number (0=none, 1=singular, 2=dual, 3=plural)
- purusa_ids: Person (0=none, 1=3rd, 2=2nd, 3=1st)
- target_root_ids: Labels for next-token prediction
- adjacency_edges: Sparse grammatical links for attention supervision

Usage:
    python scripts/gita_training_builder.py data/gita.txt -o data/gita_training.json

Note: Full morphological analysis requires vidyut-prakriya. This script uses
heuristic-based analysis for demonstration and can be enhanced with proper
morphological backends.
"""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter


# ==============================================================================
# Type Definitions (matching docs/types/data-contracts.md)
# ==============================================================================

class MorphToken(TypedDict):
    """A single morphologically analyzed token."""
    surface: str
    stem: str
    type: str  # subanta, tinanta, avyaya, krdanta, taddhita, samasa
    attributes: Dict[str, Any]


class AdjacencyEdge(TypedDict):
    """A single edge in the sparse adjacency representation."""
    src: int
    tgt: int
    link_type: str


class TrainingSample(TypedDict):
    """A single training example in factorized tensor format."""
    id: str
    chapter: int
    raw_text: str
    
    # Factorized tensor inputs (Panini-LM architecture)
    root_ids: List[int]
    type_ids: List[int]
    vibhakti_ids: List[int]
    vacana_ids: List[int]
    purusa_ids: List[int]
    
    # Training target
    target_root_ids: List[int]
    
    # Attention supervision
    adjacency_edges: List[AdjacencyEdge]
    
    # Metadata
    seq_len: int
    num_edges: int
    sparsity: float
    
    # Ground truth (for debugging)
    tokens: List[MorphToken]


class TrainingDataset(TypedDict):
    """Complete training dataset for Panini-LM."""
    metadata: Dict[str, Any]
    vocab: Dict[str, int]
    type_vocab: Dict[str, int]
    vibhakti_vocab: Dict[str, int]
    vacana_vocab: Dict[str, int]
    purusa_vocab: Dict[str, int]
    chapters: List[Dict[str, Any]]
    samples: List[TrainingSample]
    statistics: Dict[str, Any]


# ==============================================================================
# Constants
# ==============================================================================

# Special tokens
SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
    "[MASK]": 4,
}

# Token type vocabulary (Phase 1 output)
TYPE_VOCAB = {
    "subanta": 0,    # Nominal (noun/adjective/pronoun)
    "tinanta": 1,    # Finite verb
    "avyaya": 2,     # Indeclinable (particle/conjunction)
    "krdanta": 3,    # Verbal derivative (participle/infinitive)
    "taddhita": 4,   # Secondary derivative
    "samasa": 5,     # Compound
    "none": 6,       # Special tokens
}

# Vibhakti (case) vocabulary
VIBHAKTI_VOCAB = {
    "none": 0,
    "prathamā": 1,   # Nominative
    "dvitīyā": 2,    # Accusative
    "tṛtīyā": 3,     # Instrumental
    "caturthī": 4,   # Dative
    "pañcamī": 5,    # Ablative
    "ṣaṣṭhī": 6,     # Genitive
    "saptamī": 7,    # Locative
    "sambodhana": 8, # Vocative
}

# Vacana (number) vocabulary
VACANA_VOCAB = {
    "none": 0,
    "ekavacana": 1,   # Singular
    "dvivacana": 2,   # Dual
    "bahuvacana": 3,  # Plural
}

# Puruṣa (person) vocabulary
PURUSA_VOCAB = {
    "none": 0,
    "prathama": 1,  # 3rd person
    "madhyama": 2,  # 2nd person
    "uttama": 3,    # 1st person
}

# Common avyayas (indeclinables) for heuristic analysis
COMMON_AVYAYAS = {
    "च", "वा", "न", "तु", "हि", "एव", "अपि", "इव", "इति", "अथ",
    "तथा", "यथा", "किम्", "कुतः", "अतः", "तस्मात्", "यदि", "तदा", "ततः",
    "सदा", "कदा", "पुनः", "अत्र", "तत्र", "यत्र", "सर्वत्र", "कुत्र",
    "हे", "अहो", "बत", "नमः", "स्वाहा", "स्वधा", "वौषट्",
    "मा", "अलम्", "नहि", "नूनम्", "खलु", "किल", "चेत्", "यावत्",
}

# Common verb endings for heuristic detection
VERB_ENDINGS = {
    # Present tense (laṭ)
    "ति", "तः", "न्ति", "सि", "थः", "थ", "मि", "वः", "मः",
    "ते", "एते", "न्ते", "से", "आथे", "ध्वे", "ए", "वहे", "महे",
    # Imperative (loṭ)
    "तु", "ताम्", "न्तु", "हि", "तम्", "त", "आनि", "आव", "आम",
    # Past tense markers
    "त्", "यत्", "वान्", "वती", "वत्", "तवान्",
    # Future
    "ष्यति", "ष्यतः", "ष्यन्ति", "ष्यसि", "ष्यथः", "ष्यथ",
    # Optative
    "यात्", "याताम्", "युः", "याः", "यातम्", "यात", "याम्", "याव", "याम",
}

# Case ending patterns (simplified)
CASE_ENDINGS = {
    # Masculine singular patterns
    "ः": ("subanta", 1, 1),   # Nominative sing
    "म्": ("subanta", 2, 1),  # Accusative sing
    "ेन": ("subanta", 3, 1),  # Instrumental sing
    "ाय": ("subanta", 4, 1),  # Dative sing
    "ात्": ("subanta", 5, 1), # Ablative sing
    "स्य": ("subanta", 6, 1), # Genitive sing
    "े": ("subanta", 7, 1),   # Locative sing
    # Plural patterns
    "ाः": ("subanta", 1, 3),  # Nom plural masc
    "ान्": ("subanta", 2, 3), # Acc plural masc
    "ैः": ("subanta", 3, 3),  # Inst plural
    "ेभ्यः": ("subanta", 4, 3), # Dat/Abl plural
    "ाणाम्": ("subanta", 6, 3), # Gen plural
    "ेषु": ("subanta", 7, 3),   # Loc plural
    # Neuter patterns
    "ानि": ("subanta", 1, 3),  # Nom/Acc plural neut
}

# Chapter information
CHAPTER_INFO = {
    1: {"name": "Arjuna Viṣāda Yoga", "sanskrit": "अर्जुनविषादयोग"},
    2: {"name": "Sāṅkhya Yoga", "sanskrit": "साङ्ख्ययोग"},
    3: {"name": "Karma Yoga", "sanskrit": "कर्मयोग"},
    4: {"name": "Jñāna Karma Saṃnyāsa Yoga", "sanskrit": "ज्ञानकर्मसंन्यासयोग"},
    5: {"name": "Karma Saṃnyāsa Yoga", "sanskrit": "कर्मसंन्यासयोग"},
    6: {"name": "Dhyāna Yoga", "sanskrit": "ध्यानयोग"},
}


# ==============================================================================
# Heuristic Morphological Analyzer
# ==============================================================================

def analyze_token_heuristic(surface: str) -> MorphToken:
    """
    Heuristically analyze a Sanskrit token.
    
    Note: This is a simplified analyzer for demonstration. In production,
    use vidyut-prakriya or sanskrit-heritage for accurate morphological analysis.
    """
    # Clean the surface form
    surface = surface.strip()
    
    # Check for avyaya (indeclinable)
    if surface in COMMON_AVYAYAS:
        return {
            "surface": surface,
            "stem": surface,
            "type": "avyaya",
            "attributes": {}
        }
    
    # Check for verb endings
    for ending in VERB_ENDINGS:
        if surface.endswith(ending):
            stem = surface[:-len(ending)] if len(surface) > len(ending) else surface
            return {
                "surface": surface,
                "stem": stem,
                "type": "tinanta",
                "attributes": {
                    "vacana": 1,  # Default singular
                    "purusa": 1,  # Default 3rd person
                    "lakara": "lat"  # Default present
                }
            }
    
    # Check for case endings
    for ending, (token_type, vibhakti, vacana) in CASE_ENDINGS.items():
        if surface.endswith(ending):
            stem = surface[:-len(ending)] if len(surface) > len(ending) else surface
            return {
                "surface": surface,
                "stem": stem,
                "type": token_type,
                "attributes": {
                    "vibhakti": vibhakti,
                    "vacana": vacana,
                }
            }
    
    # Default: assume subanta (nominal) with nominative singular
    return {
        "surface": surface,
        "stem": surface,
        "type": "subanta",
        "attributes": {
            "vibhakti": 1,
            "vacana": 1,
        }
    }


def extract_stem(surface: str, token_type: str) -> str:
    """Extract the stem/root from a surface form."""
    # For verbs, try to find the dhātu (root)
    if token_type == "tinanta":
        for ending in VERB_ENDINGS:
            if surface.endswith(ending):
                return surface[:-len(ending)] if len(surface) > len(ending) else surface
    
    # For nominals, remove case endings
    if token_type == "subanta":
        for ending in CASE_ENDINGS:
            if surface.endswith(ending):
                return surface[:-len(ending)] if len(surface) > len(ending) else surface
    
    return surface


# ==============================================================================
# Text Processing
# ==============================================================================

def tokenize_sentence(text: str) -> List[str]:
    """
    Tokenize a Sanskrit sentence into words.
    
    Handles:
    - Devanagari text
    - Punctuation removal
    - Whitespace normalization
    """
    # Remove punctuation (keeping danda for sentence detection)
    text = re.sub(r'[।॥,;:\-—""''"\'()\[\]{}]', ' ', text)
    
    # Split on whitespace
    tokens = text.split()
    
    # Filter empty tokens
    tokens = [t.strip() for t in tokens if t.strip()]
    
    return tokens


def parse_chapters(text: str) -> List[Tuple[int, str, List[str]]]:
    """
    Parse gita.txt into chapters with sentences.
    
    Returns: List of (chapter_number, chapter_name, [sentences])
    """
    chapters = []
    current_chapter = 0
    current_name = ""
    current_sentences = []
    
    lines = text.split('\n')
    
    # Chapter header pattern: अध्याय १ (अर्जुनविषादयोग)
    chapter_pattern = re.compile(r'^अध्याय\s+([१२३४५६७८९०\d]+)\s*\(([^)]+)\)')
    divider_pattern = re.compile(r'^[—\-ー]{3,}$')
    
    # Devanagari numeral conversion
    deva_to_int = {'०': 0, '१': 1, '२': 2, '३': 3, '४': 4, 
                   '५': 5, '६': 6, '७': 7, '८': 8, '९': 9}
    
    def convert_deva_num(s: str) -> int:
        result = 0
        for c in s:
            if c in deva_to_int:
                result = result * 10 + deva_to_int[c]
            elif c.isdigit():
                result = result * 10 + int(c)
        return result
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and dividers
        if not line or divider_pattern.match(line):
            continue
        
        # Check for chapter header
        match = chapter_pattern.match(line)
        if match:
            # Save previous chapter
            if current_chapter > 0 and current_sentences:
                chapters.append((current_chapter, current_name, current_sentences))
            
            # Start new chapter
            current_chapter = convert_deva_num(match.group(1))
            current_name = match.group(2)
            current_sentences = []
            continue
        
        # Skip speaker attributions as separate sentences
        # (they'll be part of the sentence)
        if re.match(r'^(श्रीभगवान्|अर्जुनः|सञ्जयः|धृतराष्ट्रः|भगवान्)\s+उवाच\s*[—\-]?\s*$', line):
            continue
        
        # Add as sentence if we're in a chapter
        if current_chapter > 0 and line:
            # Clean up speaker attributions inline
            line = re.sub(r'^(श्रीभगवान्|अर्जुनः|सञ्जयः|धृतराष्ट्रः|भगवान्)\s+उवाच\s*[—\-]?\s*', '', line)
            if line.strip():
                current_sentences.append(line.strip())
    
    # Don't forget the last chapter
    if current_chapter > 0 and current_sentences:
        chapters.append((current_chapter, current_name, current_sentences))
    
    return chapters


# ==============================================================================
# Training Data Builder
# ==============================================================================

@dataclass
class GitaTrainingBuilder:
    """
    Build factorized tensor training data from gita.txt.
    
    Implements the Panini-LM training data specification with:
    - Factorized embeddings (root, type, vibhakti, vacana, puruṣa)
    - Sparse adjacency edges for attention supervision
    - Zero-OOV vocabulary architecture
    """
    
    # Vocabularies
    vocab: Dict[str, int] = field(default_factory=lambda: dict(SPECIAL_TOKENS))
    next_vocab_id: int = len(SPECIAL_TOKENS)
    
    # Statistics
    total_tokens: int = 0
    total_sentences: int = 0
    type_counts: Counter = field(default_factory=Counter)
    
    def get_or_add_vocab(self, stem: str) -> int:
        """Get vocab ID for a stem, adding if not present."""
        if stem not in self.vocab:
            self.vocab[stem] = self.next_vocab_id
            self.next_vocab_id += 1
        return self.vocab[stem]
    
    def analyze_sentence(self, text: str) -> List[MorphToken]:
        """Analyze a sentence into morphological tokens."""
        words = tokenize_sentence(text)
        tokens = []
        
        for word in words:
            token = analyze_token_heuristic(word)
            tokens.append(token)
            self.type_counts[token["type"]] += 1
        
        return tokens
    
    def factorize_tokens(self, tokens: List[MorphToken]) -> Dict[str, List[int]]:
        """
        Convert MorphTokens to factorized tensor representation.
        
        This is the key step for Panini-LM's factorized embedding architecture.
        """
        root_ids = [SPECIAL_TOKENS["[BOS]"]]
        type_ids = [TYPE_VOCAB["none"]]
        vibhakti_ids = [VIBHAKTI_VOCAB["none"]]
        vacana_ids = [VACANA_VOCAB["none"]]
        purusa_ids = [PURUSA_VOCAB["none"]]
        
        for token in tokens:
            # Root/stem ID
            root_id = self.get_or_add_vocab(token["stem"])
            root_ids.append(root_id)
            
            # Type ID
            type_id = TYPE_VOCAB.get(token["type"], TYPE_VOCAB["none"])
            type_ids.append(type_id)
            
            # Vibhakti (case)
            vibhakti = token["attributes"].get("vibhakti", 0)
            vibhakti_ids.append(vibhakti)
            
            # Vacana (number)
            vacana = token["attributes"].get("vacana", 0)
            vacana_ids.append(vacana)
            
            # Puruṣa (person) - only for verbs
            purusa = token["attributes"].get("purusa", 0)
            purusa_ids.append(purusa)
        
        # Add EOS
        root_ids.append(SPECIAL_TOKENS["[EOS]"])
        type_ids.append(TYPE_VOCAB["none"])
        vibhakti_ids.append(VIBHAKTI_VOCAB["none"])
        vacana_ids.append(VACANA_VOCAB["none"])
        purusa_ids.append(PURUSA_VOCAB["none"])
        
        return {
            "root_ids": root_ids,
            "type_ids": type_ids,
            "vibhakti_ids": vibhakti_ids,
            "vacana_ids": vacana_ids,
            "purusa_ids": purusa_ids,
        }
    
    def build_adjacency_edges(self, tokens: List[MorphToken]) -> List[AdjacencyEdge]:
        """
        Build sparse adjacency edges for attention supervision.
        
        Implements basic kāraka (semantic role) linking:
        - Nominals connect to verb (kartā-kriyā, karma-kriyā, etc.)
        - Adjacent tokens have local context edges
        - Determiners/adjectives connect to their head noun
        """
        edges = []
        seq_len = len(tokens) + 2  # +2 for BOS/EOS
        
        # Find verb position (main verb is typically the action center)
        verb_positions = []
        for i, token in enumerate(tokens):
            if token["type"] == "tinanta":
                verb_positions.append(i + 1)  # +1 for BOS offset
        
        # Connect nominals to verbs
        for i, token in enumerate(tokens):
            token_pos = i + 1  # +1 for BOS offset
            
            if token["type"] == "subanta":
                for verb_pos in verb_positions:
                    # Determine link type based on case
                    vibhakti = token["attributes"].get("vibhakti", 0)
                    
                    if vibhakti == 1:  # Nominative → kartā (agent)
                        link_type = "kartā-kriyā"
                    elif vibhakti == 2:  # Accusative → karma (patient)
                        link_type = "karma-kriyā"
                    elif vibhakti == 3:  # Instrumental → karaṇa
                        link_type = "karaṇa-kriyā"
                    elif vibhakti == 4:  # Dative → sampradāna
                        link_type = "sampradāna-kriyā"
                    elif vibhakti == 5:  # Ablative → apādāna
                        link_type = "apādāna-kriyā"
                    elif vibhakti == 7:  # Locative → adhikaraṇa
                        link_type = "adhikaraṇa-kriyā"
                    else:
                        link_type = "viśeṣya-viśeṣaṇa"
                    
                    edges.append({
                        "src": token_pos,
                        "tgt": verb_pos,
                        "link_type": link_type
                    })
        
        # Local context edges (adjacent tokens)
        for i in range(seq_len - 1):
            edges.append({
                "src": i,
                "tgt": i + 1,
                "link_type": "adjacent"
            })
        
        # Self-attention edges (each token attends to itself)
        for i in range(seq_len):
            edges.append({
                "src": i,
                "tgt": i,
                "link_type": "self"
            })
        
        return edges
    
    def process_sentence(self, text: str, chapter: int, sentence_idx: int) -> TrainingSample:
        """Process a single sentence into training format."""
        # Analyze tokens
        tokens = self.analyze_sentence(text)
        
        # Factorize
        factorized = self.factorize_tokens(tokens)
        
        # Build adjacency
        edges = self.build_adjacency_edges(tokens)
        
        # Create target (shifted root_ids for next-token prediction)
        target_root_ids = factorized["root_ids"][1:] + [SPECIAL_TOKENS["[PAD]"]]
        
        # Calculate sparsity
        seq_len = len(factorized["root_ids"])
        num_edges = len(edges)
        sparsity = num_edges / (seq_len * seq_len) if seq_len > 0 else 0
        
        self.total_tokens += len(tokens)
        self.total_sentences += 1
        
        return {
            "id": f"ch{chapter:02d}_s{sentence_idx:04d}",
            "chapter": chapter,
            "raw_text": text,
            
            "root_ids": factorized["root_ids"],
            "type_ids": factorized["type_ids"],
            "vibhakti_ids": factorized["vibhakti_ids"],
            "vacana_ids": factorized["vacana_ids"],
            "purusa_ids": factorized["purusa_ids"],
            
            "target_root_ids": target_root_ids,
            
            "adjacency_edges": edges,
            
            "seq_len": seq_len,
            "num_edges": num_edges,
            "sparsity": round(sparsity, 4),
            
            "tokens": tokens,
        }
    
    def build_dataset(self, gita_path: str) -> TrainingDataset:
        """Build complete training dataset from gita.txt."""
        # Read file
        with open(gita_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Parse chapters
        chapters = parse_chapters(text)
        
        # Process all sentences
        samples = []
        chapter_summaries = []
        
        for chapter_num, chapter_name, sentences in chapters:
            chapter_samples = []
            
            for idx, sentence in enumerate(sentences):
                sample = self.process_sentence(sentence, chapter_num, idx)
                chapter_samples.append(sample)
            
            samples.extend(chapter_samples)
            
            chapter_summaries.append({
                "number": chapter_num,
                "name": chapter_name,
                "sanskrit_name": CHAPTER_INFO.get(chapter_num, {}).get("sanskrit", chapter_name),
                "num_sentences": len(chapter_samples),
                "num_tokens": sum(s["seq_len"] - 2 for s in chapter_samples),  # -2 for BOS/EOS
            })
        
        # Calculate statistics
        seq_lens = [s["seq_len"] for s in samples]
        sparsities = [s["sparsity"] for s in samples]
        
        statistics = {
            "total_samples": len(samples),
            "total_tokens": self.total_tokens,
            "vocab_size": len(self.vocab),
            "unk_rate": 0.0,  # Zero OOV by design
            "seq_len": {
                "min": min(seq_lens) if seq_lens else 0,
                "max": max(seq_lens) if seq_lens else 0,
                "mean": round(sum(seq_lens) / len(seq_lens), 2) if seq_lens else 0,
            },
            "sparsity": {
                "min": round(min(sparsities), 4) if sparsities else 0,
                "max": round(max(sparsities), 4) if sparsities else 0,
                "mean": round(sum(sparsities) / len(sparsities), 4) if sparsities else 0,
            },
            "type_distribution": dict(self.type_counts),
        }
        
        return {
            "metadata": {
                "source": str(gita_path),
                "created_at": datetime.now().isoformat(),
                "panini_lm_version": "0.1.0",
                "description": "Bhagavad Gita training data for Panini-LM (factorized tensors)",
                "factorized_embeddings": True,
                "num_chapters": len(chapters),
            },
            "vocab": self.vocab,
            "type_vocab": TYPE_VOCAB,
            "vibhakti_vocab": VIBHAKTI_VOCAB,
            "vacana_vocab": VACANA_VOCAB,
            "purusa_vocab": PURUSA_VOCAB,
            "chapters": chapter_summaries,
            "samples": samples,
            "statistics": statistics,
        }
    
    def save_dataset(self, dataset: TrainingDataset, output_path: str) -> None:
        """Save dataset to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset saved to {output_path}")
        print(f"  Samples: {dataset['statistics']['total_samples']}")
        print(f"  Tokens: {dataset['statistics']['total_tokens']}")
        print(f"  Vocab size: {dataset['statistics']['vocab_size']}")


def main():
    parser = argparse.ArgumentParser(description='Build Panini-LM training data from gita.txt')
    parser.add_argument('input', help='Path to gita.txt')
    parser.add_argument('-o', '--output', default='data/gita_training.json',
                        help='Output path for training JSON')
    args = parser.parse_args()
    
    builder = GitaTrainingBuilder()
    dataset = builder.build_dataset(args.input)
    builder.save_dataset(dataset, args.output)


if __name__ == '__main__':
    main()
