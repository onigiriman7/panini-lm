#!/usr/bin/env python3
"""
Dhaatu (धातु) Text to JSON Converter for Panini-LM

Parses the structured Sanskrit verb root (dhātu) database and converts it
to a JSON format suitable for the PaninianEmbedding layer.

The JSON structure enables:
- Zero OOV: Direct lookup of root embeddings by ID
- Morphological feature extraction for factorized embeddings
- Semantic search via English/Hindi meanings
"""

import json
import re
from pathlib import Path
from typing import Optional


def parse_dhaatu_line(line: str) -> Optional[dict]:
    """
    Parse a single dhaatu entry line into structured form.
    
    Example input:
    ०१.०००१ (कौमुदीधातुः-१) भू भू सत्तायाम् भ्वादिः परस्मैपदी अकर्मकः सेट् (to exist...) (होना)
    
    Returns dict with:
    - id: Sequential ID for embedding lookup
    - reference: Original reference number (e.g., "01.0001")
    - kaumudi_ref: Kaumudi reference number
    - root: Basic root form
    - root_with_markers: Root with anubandhas
    - meaning_sanskrit: Sanskrit meaning
    - gana: Verb class (1-10)
    - pada: Voice type (parasmaipadi/atmanepadi/ubhayapadi)
    - transitivity: akarmaka/sakarmaka/dvikarmaka
    - set_type: seT/aniT/veT
    - meanings_en: List of English meanings
    - meanings_hi: List of Hindi meanings
    """
    line = line.strip()
    if not line:
        return None
    
    # Skip non-standard entries (sutras and rules at the end)
    if not re.match(r'^[०-९]+\.[०-९]+', line):
        return None
    
    entry = {}
    
    # Extract reference number and convert to decimal
    ref_match = re.match(r'^([०-९]+)\.([०-९]+)', line)
    if ref_match:
        # Convert Devanagari digits to decimal
        devanagari_to_decimal = str.maketrans('०१२३४५६७८९', '0123456789')
        ref1 = ref_match.group(1).translate(devanagari_to_decimal)
        ref2 = ref_match.group(2).translate(devanagari_to_decimal)
        entry['reference'] = f"{ref1.zfill(2)}.{ref2.zfill(4)}"
        entry['gana_num'] = int(ref1)  # Verb class from reference
        line = line[ref_match.end():].strip()
    else:
        return None
    
    # Extract Kaumudi reference
    kaumudi_match = re.match(r'\(([^)]+)\)', line)
    if kaumudi_match:
        entry['kaumudi_ref'] = kaumudi_match.group(1)
        line = line[kaumudi_match.end():].strip()
    
    # Extract Hindi meanings (last parentheses with Devanagari)
    hindi_match = re.search(r'\(([^()]*[ा-ॿ][^()]*)\)\s*$', line)
    if hindi_match:
        hindi_text = hindi_match.group(1)
        entry['meanings_hi'] = [m.strip() for m in hindi_text.split(',')]
        line = line[:hindi_match.start()].strip()
    
    # Extract English meanings (parentheses with "to ...")
    english_match = re.search(r'\(([^()]*to [^()]*)\)\s*$', line)
    if english_match:
        english_text = english_match.group(1)
        entry['meanings_en'] = [m.strip() for m in english_text.split(',')]
        line = line[:english_match.start()].strip()
    
    # Parse remaining structure: root root_markers meaning gana pada karma set
    parts = line.split()
    
    if len(parts) >= 2:
        entry['root'] = parts[0]
        entry['root_with_markers'] = parts[1]
    
    # Find gana (verb class)
    gana_map = {
        'भ्वादिः': ('bhvadi', 1),
        'अदादिः': ('adadi', 2),
        'जुहोत्यादिः': ('juhotyadi', 3),
        'दिवादिः': ('divadi', 4),
        'स्वादिः': ('svadi', 5),
        'तुदादिः': ('tudadi', 6),
        'रुधादिः': ('rudhadi', 7),
        'तनादिः': ('tanadi', 8),
        'क्र्यादिः': ('kryadi', 9),
        'चुरादिः': ('curadi', 10),
    }
    
    for gana_sk, (gana_en, gana_num) in gana_map.items():
        if gana_sk in line:
            entry['gana'] = gana_en
            entry['gana_num'] = gana_num
            break
    
    # Find pada (voice)
    pada_map = {
        'परस्मैपदी': 'parasmaipadi',
        'आत्मनेपदी': 'atmanepadi',
        'उभयपदी': 'ubhayapadi',
    }
    
    for pada_sk, pada_en in pada_map.items():
        if pada_sk in line:
            entry['pada'] = pada_en
            break
    
    # Find transitivity
    karma_map = {
        'अकर्मकः': 'akarmaka',      # intransitive
        'सकर्मकः': 'sakarmaka',      # transitive
        'द्विकर्मकः': 'dvikarmaka',  # ditransitive
    }
    
    for karma_sk, karma_en in karma_map.items():
        if karma_sk in line:
            entry['transitivity'] = karma_en
            break
    
    # Find seT/aniT/veT
    set_map = {
        'सेट्': 'seT',   # takes iT
        'अनिट्': 'aniT', # doesn't take iT
        'वेट्': 'veT',   # optionally takes iT
    }
    
    for set_sk, set_en in set_map.items():
        if set_sk in line:
            entry['set_type'] = set_en
            break
    
    # Extract Sanskrit meaning (after root_with_markers, before gana)
    # Look for the meaning section
    meaning_match = re.search(r'ँ?\s+([^\s]+(?:\s+[^\s]+)?)\s+(?:भ्वादिः|अदादिः|जुहोत्यादिः|दिवादिः|स्वादिः|तुदादिः|रुधादिः|तनादिः|क्र्यादिः|चुरादिः)', line)
    if meaning_match:
        entry['meaning_sanskrit'] = meaning_match.group(1)
    
    return entry if 'root' in entry else None


def convert_dhaatu_to_json(input_path: Path, output_path: Path):
    """
    Convert dhaatu.txt to structured JSON format.
    """
    entries = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            entry = parse_dhaatu_line(line)
            if entry:
                entry['id'] = len(entries)  # Sequential ID for embedding lookup
                entries.append(entry)
    
    # Create the final structure
    output = {
        "_metadata": {
            "description": "Sanskrit Dhātu (verb root) database for Panini-LM",
            "source": "dhaatu.txt - Based on Siddhānta-Kaumudī",
            "total_roots": len(entries),
            "gana_counts": {},
            "pada_counts": {},
            "transitivity_counts": {},
        },
        "dhatus": entries
    }
    
    # Calculate statistics
    for entry in entries:
        gana = entry.get('gana', 'unknown')
        pada = entry.get('pada', 'unknown')
        trans = entry.get('transitivity', 'unknown')
        
        output["_metadata"]["gana_counts"][gana] = output["_metadata"]["gana_counts"].get(gana, 0) + 1
        output["_metadata"]["pada_counts"][pada] = output["_metadata"]["pada_counts"].get(pada, 0) + 1
        output["_metadata"]["transitivity_counts"][trans] = output["_metadata"]["transitivity_counts"].get(trans, 0) + 1
    
    # Write JSON output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(entries)} dhātu entries to {output_path}")
    print(f"Statistics:")
    print(f"  Gaṇa distribution: {output['_metadata']['gana_counts']}")
    print(f"  Pada distribution: {output['_metadata']['pada_counts']}")
    print(f"  Transitivity: {output['_metadata']['transitivity_counts']}")
    
    return output


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "dhaatu.txt"
    output_file = project_root / "data" / "dhaatu.json"
    
    convert_dhaatu_to_json(input_file, output_file)
