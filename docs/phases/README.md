# Phases Overview

> The 5-phase processing pipeline of Panini-LM.

---

## Pipeline Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     PANINI-LM PIPELINE                           │
│                 (Factorized Embeddings Architecture)             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Raw Sanskrit Text                                              │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  PHASE 1: Morphological Ingestion + Factorization       │   │
│   │  • Sandhi resolution                                    │   │
│   │  • Samāsa decomposition                                 │   │
│   │  • Attribute extraction (vibhakti, vacana, etc.)        │   │
│   │  • FACTORIZE → root_ids, type_ids, vibhakti_ids, etc.  │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│              ┌────────────┴────────────┐                        │
│              ▼                         ▼                        │
│   ┌─────────────────────┐   ┌─────────────────────┐            │
│   │  PHASE 2A           │   │  PHASE 2B           │            │
│   │  Symbolic Engine    │   │  Neural Engine      │            │
│   │  ────────────────   │   │  ────────────────   │            │
│   │  • Kāraka mapping   │   │  • FACTORIZED       │            │
│   │  • Rule evaluation  │   │    EMBEDDING        │            │
│   │  • Matrix M (N×N)   │   │  • E = Σ components │            │
│   │                     │   │  • Q, K, V proj     │            │
│   │  Uses: MorphTokens  │   │  Uses: Factorized   │            │
│   │  (attributes)       │   │  ID tensors         │            │
│   └──────────┬──────────┘   └──────────┬──────────┘            │
│              │                         │                        │
│              └────────────┬────────────┘                        │
│                           ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  PHASE 3: Sparse Pāṇinian Attention                     │   │
│   │  • O(N·k) complexity (vs O(N²))                         │   │
│   │  • Grammatical routing via Matrix M                     │   │
│   │  • Q,K,V from factorized embeddings                     │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  PHASE 4: Semantic Maturation                           │   │
│   │  • SwiGLU Feed-Forward Network                          │   │
│   │  • Reduced expansion (1.5-2x vs 4x)                     │   │
│   │  • Pure semantic processing (grammar externalized)      │   │
│   └───────────────────────┬─────────────────────────────────┘   │
│                           ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  PHASE 5: Grammar-Constrained Decoding                  │   │
│   │  • Output over ~4000 roots (not 50000 surface forms)    │   │
│   │  • Mask invalid roots (P = 0)                          │   │
│   │  • 100% grammatical correctness guarantee               │   │
│   │  • Reconstruct surface form via morphology              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│   Generated Sanskrit Text (grammatically perfect)                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase Summary

| Phase | Name | Input | Output | Key Innovation |
|-------|------|-------|--------|----------------|
| **1** | [Morphological Ingestion](phase1-morphology.md) | Raw UTF-8 text | `MorphTokens` + `FactorizedTokenBatch` | FST-based Sandhi/Samāsa + Factorization |
| **2A** | [Symbolic Engine](phase2a-symbolic.md) | `List[MorphToken]` | Matrix M `(N×N)` | Deterministic grammatical routing |
| **2B** | [Neural Engine](phase2b-neural.md) | `FactorizedTokenBatch` | Q, K, V tensors | **Factorized embeddings** (~4000 vocab) |
| **3** | [Sparse Attention](phase3-attention.md) | Q, K, V, M | Hidden states | O(N·k) hardware-optimized routing |
| **4** | [Semantic Maturation](phase4-ffn.md) | Hidden states | Refined states | Compact FFN (1.5-2x expansion) |
| **5** | [Grammar Decoding](phase5-decoding.md) | Logits + state | Valid tokens | 100% grammatical + Zero OOV |

---

## The Factorized Embedding Breakthrough

### Vocabulary Comparison

| Approach | Vocabulary | Embedding Params |
|----------|------------|------------------|
| Standard Transformer | 50,000+ surface forms | 25.6M |
| **Panini-LM** | ~4,000 morphological primitives | **2.06M** |

### How Factorization Works

Instead of one embedding per inflected word, Panini-LM constructs embeddings by summing components:

```python
E(gacchati) = E(√gam)      # root embedding (~4000 options)
            + E(tiṅanta)    # type embedding (7 options)
            + E(laṭ)        # tense/mood (encoded in type)
            + E(prathama)   # person embedding (4 options)
            + E(eka-vacana) # number embedding (4 options)
```

### Benefits

1. **Zero OOV**: Any valid inflection can be embedded, even if never seen
2. **12× Parameter Reduction**: ~2M vs ~25M embedding parameters
3. **Structural Encoding**: Morphological knowledge is preserved, not learned

---

## Dependencies

```
Phase 1 ──┬──► Phase 2A ──┐
          │   (tokens)    │
          │               ├──► Phase 3 ──► Phase 4 ──► Phase 5
          │               │                              │
          └──► Phase 2B ──┘                              │
            (factorized)                                 │
                                                         │
          Phase 1 (morph engine) ◄───────────────────────┘
                    (called at inference for grammar mask
                     and surface form reconstruction)
```

### Parallel Execution

- **Phase 2A and 2B** run in parallel after Phase 1
- **Phase 3** requires both Phase 2A (Matrix M) and Phase 2B (Q, K, V)
- **Phase 5** requires Phase 1's morphological engine at inference time

## Data Flow Types

See [data-contracts.md](../types/data-contracts.md) for complete type definitions.

| Phase Transition | Data Type | Shape |
|------------------|-----------|-------|
| Input → Phase 1 | `str` | Variable length |
| Phase 1 → Phase 2A | `List[MorphToken]` | N tokens |
| Phase 1 → Phase 2B | `List[MorphToken]` | N tokens |
| Phase 2A → Phase 3 | `torch.Tensor` (M) | (N, N) |
| Phase 2B → Phase 3 | `QKVTensors` | (batch, heads, N, head_dim) |
| Phase 3 → Phase 4 | `torch.Tensor` | (batch, N, d_model) |
| Phase 4 → Phase 5 | `torch.Tensor` | (batch, N, d_model) |
| Phase 5 → Output | `str` | Variable length |

---

## Performance Characteristics

### Computational Complexity

| Phase | Complexity | Notes |
|-------|------------|-------|
| Phase 1 | O(N) | Dictionary lookup per token |
| Phase 2A | O(N²) | Rule evaluation, but deterministic and parallelizable |
| Phase 2B | O(N·d) | Linear projections |
| Phase 3 | **O(N·k)** | Key innovation: k ≈ 2-3 << N |
| Phase 4 | O(N·d²) | Standard FFN |
| Phase 5 | O(V) | Mask generation per vocab |

### Memory Usage

| Operation | Standard Transformer | Panini-LM | Reduction |
|-----------|---------------------|-----------|-----------|
| Attention scores | O(N²) | O(N·k) | **~95%** |
| Total model params | 100% | 70-80% | **20-30%** |

---

## Related Documentation

- [Data Contracts](../types/data-contracts.md) — Type definitions
- [Test Specifications](../testing/test-specifications.md) — Testing
- [Integration Guide](../integration/README.md) — External libraries
- [Glossary](../GLOSSARY.md) — Terminology
