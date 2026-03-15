# Panini-LM: Neuro-Symbolic Language Modeling via Pāṇinian Algebraic Priors

> **Quick navigation**: [INDEX](INDEX.md) — [GLOSSARY](GLOSSARY.md) — [Data Types](types/data-contracts.md) — [Training](training/README.md) — [Tests](testing/test-specifications.md)

---

## Overview

This repository documents the Pāṇinian Neuro-Symbolic architecture for Sanskrit language modeling. The documentation is organized by architecture phases.

### Phase Documents

| Phase | Document | Description |
|-------|----------|-------------|
| 1 | [Morphological Ingestion](phases/phase1-morphology.md) | Sandhi/Samāsa resolution, token extraction |
| 2A | [Symbolic Engine](phases/phase2a-symbolic.md) | Deterministic adjacency matrix M |
| 2B | [Neural Engine](phases/phase2b-neural.md) | Position-agnostic Q, K, V embeddings |
| 3 | [Sparse Attention](phases/phase3-attention.md) | O(N·k) grammatical routing |
| 4 | [FFN / Maturation](phases/phase4-ffn.md) | Compact SwiGLU (1.5-2x expansion) |
| 5 | [Constrained Decoding](phases/phase5-decoding.md) | 100% grammatical output guarantee |

### Integration Guides

- [Integration Overview](integration/README.md) — External library summary
- [vidyut-prakriya](integration/vidyut.md) — Primary morphological backend
- [sanskrit-heritage](integration/sanskrit-heritage.md) — Fallback implementation
- [Triton Kernels](integration/triton.md) — GPU sparse attention
- [samsadhani](integration/samsadhani.md) — Kāraka analysis API

### Training

- [Training Guide](training/README.md) — Dataset structure, training process, model size

---

## Model Size

Panini-LM is designed to be parameter-efficient by offloading syntax to symbolic rules:

| Config | Parameters | Notes |
|--------|------------|-------|
| **Small** | ~15M | d_model=256, 4 layers |
| **Default** | ~39M | d_model=512, 6 layers |
| **Base** | ~85M | d_model=768, 8 layers |
| **Large** | ~180M | d_model=1024, 12 layers |

**Why smaller than GPT/BERT?**
- No positional encoding (Sanskrit has free word order)
- Reduced FFN expansion (1.5× vs 4×) — syntax handled by Phase 2A
- Domain-specific vocabulary (Sanskrit only)

---

## Architecture Summary

Current Large Language Models (LLMs) rely on dense parameter networks and autoregressive attention mechanisms to statistically approximate syntactic structures. For morphologically rich, free-word-order languages like Sanskrit, this results in extreme computational inefficiencies, high "token taxes," and arbitrary positional dependencies.

We propose a novel Neuro-Symbolic Transformer architecture that decouples syntax from semantics. By mapping the generative rules of Pāṇini's *Aṣṭādhyāyī* to deterministic mathematical functions, we generate a highly sparse, mathematically verified Adjacency Matrix ($M$). Injecting this matrix directly into the attention mechanism as a structural prior forces the network to route purely along valid *Kāraka* (dependency) pathways.

This architecture fundamentally reduces the computational complexity of syntactic attention from $O(N^2)$ to $O(N \cdot k)$, freeing the neural parameters to model pure semantics and exponentially reducing the data required for convergence.

## Key Innovations

- **Symbolic Syntax Processing**: Complete offloading of syntax to deterministic rule-based engines
- **Sparse Attention Mechanism**: Hardware-optimized routing using grammatical adjacency matrices
- **Position-Agnostic Embeddings**: Native support for free-word-order languages
- **Grammar-Constrained Decoding**: Elimination of grammatical hallucinations during generation

## Repository Structure

```
panini-lm/
├── docs/
│   ├── INDEX.md              # Central navigation
│   ├── GLOSSARY.md           # Sanskrit & architecture terms
│   ├── Overview.md           # This file
│   ├── phases/               # Canonical phase documentation
│   │   ├── README.md
│   │   ├── phase1-morphology.md
│   │   ├── phase2a-symbolic.md
│   │   ├── phase2b-neural.md
│   │   ├── phase3-attention.md
│   │   ├── phase4-ffn.md
│   │   └── phase5-decoding.md
│   ├── integration/          # External library guides
│   │   ├── README.md
│   │   ├── vidyut.md
│   │   ├── sanskrit-heritage.md
│   │   ├── triton.md
│   │   └── samsadhani.md
│   ├── types/                # Data contracts
│   │   └── data-contracts.md
│   └── testing/              # Test specifications
│       └── test-specifications.md
├── pseudo-code.py            # PyTorch implementation reference
├── external-module.txt       # External integration blueprint
└── README.md                 # Project overview
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/onigiriman7/panini-lm.git
cd panini-lm

# Install dependencies
pip install torch vidyut_py triton

# Run the model
python pseudo-code.py
```

## Further Reading

| Topic | Document |
|-------|----------|
| Full architecture | [INDEX.md](INDEX.md) |
| Terminology | [GLOSSARY.md](GLOSSARY.md) |
| Data structures | [types/data-contracts.md](types/data-contracts.md) |
| Testing | [testing/test-specifications.md](testing/test-specifications.md) |
| Legacy architecture | [general/Architecture.md](general/Architecture.md) |
| Legacy efficiency | [general/Efficiency.md](general/Efficiency.md) |