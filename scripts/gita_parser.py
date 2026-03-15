#!/usr/bin/env python3
"""
Gita Parser — Extract chapters and sentences from gita.txt.

This module parses the Bhagavad Gita prose form into structured chapters
and sentences suitable for the Panini-LM training pipeline.

Usage:
    from scripts.gita_parser import GitaParser
    
    parser = GitaParser()
    chapters = parser.parse_file("gita.txt")
    
    for chapter in chapters:
        print(f"Chapter {chapter['number']}: {chapter['name']}")
        for sentence in chapter['sentences']:
            print(f"  - {sentence[:50]}...")
"""

import re
from pathlib import Path
from typing import List, Dict, TypedDict, Optional
from dataclasses import dataclass, field


class Sentence(TypedDict):
    """A single sentence with metadata."""
    text: str
    chapter: int
    line_start: int
    line_end: int


class Chapter(TypedDict):
    """A chapter with sentences."""
    number: int
    name: str
    sanskrit_name: str
    line_start: int
    line_end: int
    sentences: List[Sentence]


# Manual chapter mapping for chapters without explicit headers
# Derived from content analysis of gita.txt
CHAPTER_MAPPINGS = {
    1: {"name": "Arjuna Visada Yoga", "sanskrit": "अर्जुनविषादयोग"},
    2: {"name": "Sankhya Yoga", "sanskrit": "साङ्ख्ययोग"},
    3: {"name": "Karma Yoga", "sanskrit": "कर्मयोग"},
    4: {"name": "Jnana Karma Sannyasa Yoga", "sanskrit": "ज्ञानकर्मसंन्यासयोग"},
    5: {"name": "Karma Sannyasa Yoga", "sanskrit": "कर्मसंन्यासयोग"},
    6: {"name": "Dhyana Yoga", "sanskrit": "ध्यानयोग"},
    7: {"name": "Jnana Vijnana Yoga", "sanskrit": "ज्ञानविज्ञानयोग"},
    8: {"name": "Aksara Brahma Yoga", "sanskrit": "अक्षरब्रह्मयोग"},
    9: {"name": "Rajavidya Rajaguhya Yoga", "sanskrit": "राजविद्याराजगुह्ययोग"},
    10: {"name": "Vibhuti Yoga", "sanskrit": "विभूतियोग"},
    11: {"name": "Visvarupa Darsana Yoga", "sanskrit": "विश्वरूपदर्शनयोग"},
    12: {"name": "Bhakti Yoga", "sanskrit": "भक्तियोग"},
    13: {"name": "Ksetra Ksetrajna Vibhaga Yoga", "sanskrit": "क्षेत्रक्षेत्रज्ञविभागयोग"},
    14: {"name": "Gunatraya Vibhaga Yoga", "sanskrit": "गुणत्रयविभागयोग"},
    15: {"name": "Purusottama Yoga", "sanskrit": "पुरुषोत्तमयोग"},
    16: {"name": "Daivasura Sampad Vibhaga Yoga", "sanskrit": "दैवासुरसंपद्विभागयोग"},
    17: {"name": "Sraddhatraya Vibhaga Yoga", "sanskrit": "श्रद्धात्रयविभागयोग"},
    18: {"name": "Moksa Sannyasa Yoga", "sanskrit": "मोक्षसंन्यासयोग"},
}


@dataclass
class GitaParser:
    """
    Parser for Bhagavad Gita prose text.
    
    Handles:
    - Explicit chapter headers (अध्याय १)
    - Divider lines (—---------) 
    - Speaker changes (श्रीभगवान् उवाच)
    - Sentence splitting (using । or period)
    """
    
    # Regex patterns
    CHAPTER_HEADER = re.compile(r'^अध्याय\s+([१२३४५६७८९०\d]+)\s*\(([^)]+)\)')
    DIVIDER_LINE = re.compile(r'^[—\-ー]{3,}$')
    SPEAKER_PATTERN = re.compile(r'^(श्रीभगवान्|अर्जुनः|सञ्जयः|धृतराष्ट्रः)\s+उवाच')
    SENTENCE_END = re.compile(r'[।|\.॥]')
    
    # Devanagari numeral map
    DEVA_NUMERALS = {'०': 0, '१': 1, '२': 2, '३': 3, '४': 4, 
                    '५': 5, '६': 6, '७': 7, '८': 8, '९': 9}
    
    def _deva_to_int(self, s: str) -> int:
        """Convert Devanagari numeral string to integer."""
        result = 0
        for char in s:
            if char in self.DEVA_NUMERALS:
                result = result * 10 + self.DEVA_NUMERALS[char]
            elif char.isdigit():
                result = result * 10 + int(char)
        return result
    
    def _split_sentences(self, text: str, chapter: int, line_num: int) -> List[Sentence]:
        """Split text into sentences using punctuation markers."""
        sentences = []
        
        # Split by sentence-ending punctuation
        parts = self.SENTENCE_END.split(text)
        
        for part in parts:
            part = part.strip()
            if part and len(part) > 2:  # Skip very short fragments
                sentences.append({
                    "text": part,
                    "chapter": chapter,
                    "line_start": line_num,
                    "line_end": line_num,
                })
        
        return sentences
    
    def _detect_chapter_boundary(
        self, 
        line: str, 
        line_num: int, 
        current_chapter: int
    ) -> Optional[int]:
        """
        Detect if a line marks a chapter boundary.
        
        Returns new chapter number or None if not a boundary.
        """
        line = line.strip()
        
        # Check explicit header
        match = self.CHAPTER_HEADER.match(line)
        if match:
            return self._deva_to_int(match.group(1))
        
        # Check divider line
        if self.DIVIDER_LINE.match(line):
            return current_chapter + 1
        
        return None
    
    def _is_duplicate_content(self, lines: List[str], start: int, length: int = 20) -> bool:
        """Check if content starting at `start` is a duplicate of earlier content."""
        if start < length * 2:
            return False
        
        snippet = ''.join(lines[start:start+length])
        earlier = ''.join(lines[:start])
        
        return snippet in earlier
    
    def parse_file(self, filepath: str) -> List[Chapter]:
        """
        Parse gita.txt and extract all chapters.
        
        Args:
            filepath: Path to gita.txt
            
        Returns:
            List of Chapter dictionaries with sentences
        """
        path = Path(filepath)
        lines = path.read_text(encoding='utf-8').splitlines()
        
        chapters: List[Chapter] = []
        current_chapter: Optional[Chapter] = None
        current_chapter_num = 0
        seen_chapters = set()  # Track which chapter numbers we've seen
        
        for line_num, line in enumerate(lines, start=1):
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Check for chapter boundary
            new_chapter_num = self._detect_chapter_boundary(
                line_stripped, line_num, current_chapter_num
            )
            
            if new_chapter_num is not None:
                # Check if this is a repeated chapter header for same chapter
                match = self.CHAPTER_HEADER.match(line_stripped)
                if match:
                    detected_num = self._deva_to_int(match.group(1))
                    if detected_num in seen_chapters:
                        # Skip duplicate chapter header - don't start new chapter
                        continue
                
                # Only start new chapter if divider leads to new content
                if self.DIVIDER_LINE.match(line_stripped):
                    # For dividers, only increment if we have content in current chapter
                    if current_chapter is not None and len(current_chapter["sentences"]) == 0:
                        # Empty chapter from previous divider, skip this divider
                        continue
                
                # Save current chapter if it has content
                if current_chapter is not None:
                    if len(current_chapter["sentences"]) > 0:
                        current_chapter["line_end"] = line_num - 1
                        chapters.append(current_chapter)
                        seen_chapters.add(current_chapter_num)
                
                # Start new chapter
                current_chapter_num = new_chapter_num
                
                # Clamp to valid chapter numbers
                if current_chapter_num > 18:
                    current_chapter_num = 18
                
                mapping = CHAPTER_MAPPINGS.get(current_chapter_num, {
                    "name": f"Chapter {current_chapter_num}",
                    "sanskrit": ""
                })
                
                current_chapter = Chapter(
                    number=current_chapter_num,
                    name=mapping["name"],
                    sanskrit_name=mapping["sanskrit"],
                    line_start=line_num,
                    line_end=line_num,
                    sentences=[],
                )
                
                # If explicit header or divider, don't treat the line as content
                if self.CHAPTER_HEADER.match(line_stripped):
                    continue
                if self.DIVIDER_LINE.match(line_stripped):
                    continue
            
            # Process content line
            if current_chapter is not None:
                sentences = self._split_sentences(
                    line_stripped, 
                    current_chapter_num, 
                    line_num
                )
                current_chapter["sentences"].extend(sentences)
        
        # Save final chapter if it has content
        if current_chapter is not None and len(current_chapter["sentences"]) > 0:
            current_chapter["line_end"] = len(lines)
            chapters.append(current_chapter)
        
        # Merge any duplicate chapter numbers (consolidate sentences)
        return self._merge_chapters(chapters)
    
    def get_all_sentences(self, filepath: str) -> List[Sentence]:
        """Get all sentences across all chapters."""
        chapters = self.parse_file(filepath)
        all_sentences = []
        for chapter in chapters:
            all_sentences.extend(chapter["sentences"])
        return all_sentences
    
    def _merge_chapters(self, chapters: List[Chapter]) -> List[Chapter]:
        """Merge chapters with same number, keeping all sentences."""
        merged = {}
        for ch in chapters:
            num = ch["number"]
            if num not in merged:
                merged[num] = Chapter(
                    number=num,
                    name=ch["name"],
                    sanskrit_name=ch["sanskrit_name"],
                    line_start=ch["line_start"],
                    line_end=ch["line_end"],
                    sentences=list(ch["sentences"]),
                )
            else:
                # Extend with additional sentences
                merged[num]["sentences"].extend(ch["sentences"])
                merged[num]["line_end"] = max(merged[num]["line_end"], ch["line_end"])
        
        # Return sorted by chapter number
        return [merged[k] for k in sorted(merged.keys())]
    
    def get_chapter_summary(self, chapters: List[Chapter]) -> str:
        """Generate a summary of parsed chapters."""
        lines = ["=" * 60, "Bhagavad Gita Chapter Summary", "=" * 60]
        
        total_sentences = 0
        for ch in chapters:
            n_sent = len(ch["sentences"])
            total_sentences += n_sent
            lines.append(
                f"Ch {ch['number']:2d} | {ch['sanskrit_name'][:25]:<25} | "
                f"Lines {ch['line_start']:4d}-{ch['line_end']:4d} | "
                f"{n_sent:3d} sentences"
            )
        
        lines.append("=" * 60)
        lines.append(f"Total: {len(chapters)} chapters, {total_sentences} sentences")
        
        return "\n".join(lines)


def main():
    """CLI entry point."""
    import sys
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else "gita.txt"
    
    parser = GitaParser()
    chapters = parser.parse_file(filepath)
    
    print(parser.get_chapter_summary(chapters))
    
    # Show sample sentences
    print("\n" + "=" * 60)
    print("Sample sentences from Chapter 2:")
    print("=" * 60)
    
    for ch in chapters:
        if ch["number"] == 2:
            for sent in ch["sentences"][:5]:
                print(f"  • {sent['text'][:80]}...")
            break


if __name__ == "__main__":
    main()
