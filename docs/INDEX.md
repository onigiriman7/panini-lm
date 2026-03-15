# Panini-LM Documentation Index

> Central navigation hub for all Panini-LM documentation.  
> **Last updated**: 2026-03-11

---

## Quick Navigation

| Category | Description | Entry Point |
|----------|-------------|-------------|
| [Overview](#overview) | Architecture concepts and key innovations | [Overview](overview/README.md) |
| [Phases](#phases) | 5-phase processing pipeline details | [Phases Overview](phases/README.md) |
| [Integration](#integration) | External library integration guides | [Integration Guide](integration/README.md) |
| [Types](#types) | Data contracts and type definitions | [Data Contracts](types/data-contracts.md) |
| [Testing](#testing) | Test specifications and validation | [Test Specs](testing/test-specifications.md) |
| [Reference](#reference) | Pseudocode and efficiency analysis | [Reference](reference/pseudocode.md) |
| [Glossary](#glossary) | Sanskrit and technical terminology | [Glossary](GLOSSARY.md) |

---

## Overview

High-level architecture documentation for the Pāṇinian Neuro-Symbolic LLM.

- [Architecture Overview](overview/README.md) — Core concepts, dual-track architecture, key innovations
- [Efficiency Analysis](reference/efficiency.md) — Computational complexity, performance benchmarks

### Key Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                    Panini-LM Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│  Raw Sanskrit Text                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                            │
│  │ Phase 1         │  Morphological Ingestion                   │
│  │ Sandhi/Samāsa   │  vidyut-prakriya / sanskrit-heritage       │
│  └────────┬────────┘                                            │
│           │                                                     │
│     ┌─────┴─────┐                                               │
│     ▼           ▼                                               │
│  ┌──────────┐  ┌──────────┐                                     │
│  │ Phase 2A │  │ Phase 2B │                                     │
│  │ Symbolic │  │ Neural   │                                     │
│  │ Matrix M │  │ Q, K, V  │                                     │
│  └────┬─────┘  └────┬─────┘                                     │
│       │             │                                           │
│       └──────┬──────┘                                           │
│              ▼                                                  │
│       ┌─────────────┐                                           │
│       │  Phase 3    │  Sparse Pāṇinian Attention                │
│       │  Attention  │  O(N·k) complexity                        │
│       └──────┬──────┘                                           │
│              ▼                                                  │
│       ┌─────────────┐                                           │
│       │  Phase 4    │  Semantic Maturation (FFN)                │
│       │  FFN        │  SwiGLU, 1.5-2x expansion                 │
│       └──────┬──────┘                                           │
│              ▼                                                  │
│       ┌─────────────┐                                           │
│       │  Phase 5    │  Grammar-Constrained Decoding             │
│       │  Decoding   │  100% grammatical correctness             │
│       └─────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phases

The 5-phase processing pipeline transforms raw Sanskrit text into grammatically-correct output.

| Phase | Name | Input | Output | Docs |
|-------|------|-------|--------|------|
| 1 | Morphological Ingestion | Raw UTF-8 text | `List[MorphToken]` | [phase1-morphology.md](phases/phase1-morphology.md) |
| 2A | Symbolic Engine | `List[MorphToken]` | Adjacency Matrix `M` | [phase2a-symbolic.md](phases/phase2a-symbolic.md) |
| 2B | Neural Engine | `FactorizedTokenBatch` | Q, K, V tensors | [phase2b-neural.md](phases/phase2b-neural.md) |
| 3 | Sparse Attention | Q, K, V, M | Contextualized states | [phase3-attention.md](phases/phase3-attention.md) |
| 4 | Semantic Maturation | Hidden states | Refined states | [phase4-ffn.md](phases/phase4-ffn.md) |
| 5 | Grammar-Constrained Decoding | Logits + morph state | Valid tokens | [phase5-decoding.md](phases/phase5-decoding.md) |

### Phase Dependencies

```
Phase 1 ──┬──► Phase 2A ──┐
          │               ├──► Phase 3 ──► Phase 4 ──► Phase 5
          └──► Phase 2B ──┘                              │
                                                         │
          Phase 1 (morph engine) ◄───────────────────────┘
```

---

## Integration

External library integration guides for each component.

| Library | Phase | Purpose | Docs |
|---------|-------|---------|------|
| vidyut-prakriya | 1 | Production morphological analysis (Rust/PyO3) | [vidyut.md](integration/vidyut.md) |
| sanskrit-heritage | 1 | Fallback morphological analysis (Python) | [sanskrit-heritage.md](integration/sanskrit-heritage.md) |
| samsadhani | 2A | Kāraka relationship API | [samsadhani.md](integration/samsadhani.md) |
| Triton | 3 | Block-sparse attention kernel | [triton.md](integration/triton.md) |
| PyTorch | 2B-5 | Neural components | [README.md](integration/README.md) |

---

## Types

Formal data contract definitions using Python TypedDict.

- [Data Contracts](types/data-contracts.md) — All inter-phase data structures
  - `MorphToken` — Phase 1 output
  - `MorphAttributes` — Morphological attributes
  - `FactorizedTokenBatch` — Phase 2B input (5 parallel ID tensors)
  - `AdjacencyMatrix` — Phase 2A output (sparse tensor)
  - Grammatical ID mappings (type, vibhakti, vacana, puruṣa)
  - Error types and validation schemas

---

## Training

Model training documentation and dataset specifications.

- [Training Guide](training/README.md) — Complete training documentation
  - **Model Size**: Parameter counts for Small (~8M), Default (~18M), Base (~45M), Large (~100M)
  - **Factorized Embeddings**: 12× parameter reduction via morphological composition
  - **Dataset Structure**: JSON format with factorized tensors (root_ids, type_ids, vibhakti_ids, vacana_ids, purusa_ids)
  - **Training Process**: Loss functions, optimizer config, DataLoader setup
  - **Data Preparation**: Using TrainingDataBuilder to process Sanskrit text

### Dataset Files

| File | Samples | Purpose |
|------|---------|---------|
| `tests/data/gita_training.json` | 1,242 | Full training dataset |
| `tests/data/gita_samples.json` | 10 | Quick testing subset |

---

## Testing

Test specifications for each phase and integration tests.

- [Test Specifications](testing/test-specifications.md)
  - Unit tests per phase
  - Integration tests
  - Performance benchmarks
  - Validation criteria

---

## Reference

Implementation references and analysis documents.

- [Pseudocode](reference/pseudocode.md) — PyTorch reference implementation
- [Efficiency Analysis](reference/efficiency.md) — Computational complexity analysis

---

## Glossary

See [GLOSSARY.md](GLOSSARY.md) for definitions of:

- **Sanskrit Grammar Terms**: Kāraka, Vibhakti, Puruṣa, Vacana, Sandhi, Samāsa, etc.
- **Architecture Terms**: Sparse attention, adjacency matrix, grammar-constrained decoding
- **Library Terms**: vidyut-prakriya, sanskrit-heritage, samsadhani

---

## Keyword Index

Quick lookup for common terms (alphabetical):

| Term | Definition | Relevant Docs |
|------|------------|---------------|
| Adjacency Matrix M | Sparse (N×N) tensor encoding grammatical validity | [phase2a-symbolic.md](phases/phase2a-symbolic.md), [Data Contracts](types/data-contracts.md) |
| Aṣṭādhyāyī | Pāṇini's Sanskrit grammar treatise (~4th c. BCE) | [GLOSSARY.md](GLOSSARY.md) |
| Block-sparse | GPU kernel optimization skipping invalid blocks | [triton.md](integration/triton.md) |
| Factorized Embedding | Additive composition of morphological primitives | [phase2b-neural.md](phases/phase2b-neural.md), [Data Contracts](types/data-contracts.md) |
| FactorizedTokenBatch | 5 parallel ID tensors (root, type, vibhakti, vacana, puruṣa) | [Data Contracts](types/data-contracts.md) |
| Grammar-constrained | Decoding that guarantees grammatical correctness | [phase5-decoding.md](phases/phase5-decoding.md) |
| Kāraka | Semantic/syntactic role (agent, object, etc.) | [GLOSSARY.md](GLOSSARY.md), [phase2a-symbolic.md](phases/phase2a-symbolic.md) |
| MorphToken | Standardized morphological token structure | [Data Contracts](types/data-contracts.md) |
| Position-agnostic | Embeddings without positional encoding | [phase2b-neural.md](phases/phase2b-neural.md) |
| Puruṣa | Grammatical person (1st/2nd/3rd) | [GLOSSARY.md](GLOSSARY.md) |
| Sandhi | Euphonic combination of sounds at word boundaries | [GLOSSARY.md](GLOSSARY.md), [phase1-morphology.md](phases/phase1-morphology.md) |
| Samāsa | Compound word formation | [GLOSSARY.md](GLOSSARY.md), [phase1-morphology.md](phases/phase1-morphology.md) |
| Sparse attention | O(N·k) attention using grammatical routing | [phase3-attention.md](phases/phase3-attention.md) |
| Subanta | Nominal (noun/adjective) word form | [GLOSSARY.md](GLOSSARY.md) |
| Tiṅanta | Verbal (verb) word form | [GLOSSARY.md](GLOSSARY.md) |
| Vacana | Grammatical number (singular/dual/plural) | [GLOSSARY.md](GLOSSARY.md) |
| Vibhakti | Case ending (nominative, accusative, etc.) | [GLOSSARY.md](GLOSSARY.md) |
| vidyut-prakriya | Rust morphological analyzer | [vidyut.md](integration/vidyut.md) |
| Zero OOV | No out-of-vocabulary errors for valid inflections | [phase2b-neural.md](phases/phase2b-neural.md), [Training](training/README.md) |

---

## File Structure

```
docs/
├── INDEX.md                    ← You are here
├── GLOSSARY.md                 # Terminology definitions
├── overview/
│   └── README.md               # Architecture overview
├── phases/
│   ├── README.md               # Phase summary
│   ├── phase1-morphology.md
│   ├── phase2a-symbolic.md
│   ├── phase2b-neural.md
│   ├── phase3-attention.md
│   ├── phase4-ffn.md
│   └── phase5-decoding.md
├── integration/
│   ├── README.md               # Integration overview
│   ├── vidyut.md
│   ├── sanskrit-heritage.md
│   ├── samsadhani.md
│   └── triton.md
├── types/
│   └── data-contracts.md       # Type definitions
├── testing/
│   └── test-specifications.md  # Test specs
└── reference/
    ├── pseudocode.md
    └── efficiency.md
```

---

## Related Files

- [/external-module.txt](/external-module.txt) — Engineering blueprint (quick reference)
- [/pseudo-code.py](/pseudo-code.py) — PyTorch implementation sketch
- [/README.md](/README.md) — Project overview and setup

---

## Contributing

When adding documentation:

1. Follow the [unified format](#unified-document-format) below
2. Add entries to this INDEX.md
3. Add terminology to [GLOSSARY.md](GLOSSARY.md)
4. Update type definitions in [data-contracts.md](types/data-contracts.md) if adding new data structures

### Unified Document Format

All phase and integration documents should follow this structure:

```markdown
# [Title]

## Overview
Brief description and purpose.

## Input/Output Contract
- **Input**: Type and description
- **Output**: Type and description
- **Errors**: Possible error conditions

## Dependencies
- List of required modules/phases

## Implementation Details
Technical details and pseudocode.

## Error Handling
How errors are detected and handled.

## Test Specifications
- Test cases
- Expected values
- Validation commands

## Related Documents
Links to related docs.
```
