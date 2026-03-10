# Architecture Overview

## The Pāṇinian Neuro-Symbolic Architecture

Panini-LM implements a dual-track processing paradigm that separates syntactic analysis from semantic understanding, allowing each component to operate at peak efficiency.

### Core Design Principles

1. **Syntax-Semantics Decoupling**: Syntax is handled by deterministic algorithms, while semantics is processed by neural networks
2. **Mathematical Ground Truth**: Linguistic rules are encoded as verifiable mathematical functions
3. **Hardware-Optimized Sparsity**: Attention computation is routed through sparse grammatical pathways
4. **Order Independence**: Native support for free-word-order languages without positional encodings

### Processing Pipeline

```
Raw Sanskrit Text
       ↓
Morphological Ingestion (FST)
       ↓
┌─────────────────┬─────────────────┐
│ Symbolic Track  │  Neural Track   │
│ (Syntax)        │  (Semantics)    │
│                 │                 │
│ Attribute       │ Position-       │
│ Extraction      │ Agnostic        │
│                 │ Embeddings      │
│ Matrix          │                 │
│ Generation      │ Q/K/V           │
│                 │ Projections     │
└─────────────────┴─────────────────┘
       ↓
Sparse Pāṇinian Attention
       ↓
Semantic Maturation (FFN)
       ↓
Grammar-Constrained Decoding
```

## Component Details

### Phase 1: Morphological Ingestion

**Input**: Raw Sanskrit text string
**Output**: Pure morphological tokens with resolved Sandhi and Samasa

**Process**:
- Finite State Transducer (FST) resolves euphonic combinations (*Sandhi*)
- Compound words (*Samāsa*) are decomposed into constituent morphemes
- Morphological integrity is preserved before vectorization

**Benefits**:
- Eliminates tokenization artifacts that break morphological roots
- Maintains grammatical relationships at the morpheme level
- Enables precise syntactic analysis

### Phase 2A: Symbolic Engine (Syntax Track)

**Input**: Morphological tokens with metadata
**Output**: Sparse Adjacency Matrix $M \in \mathbb{R}^{N \times N}$

**Components**:
1. **Attribute Extraction**: Grammatical metadata tagging
2. **Rule Evaluation**: Pāṇinian algorithm application
3. **Matrix Generation**: Valid dependency pathway mapping

### Phase 2B: Neural Engine (Semantic Track)

**Input**: Token sequences
**Output**: Dense vector representations

**Features**:
- Position-agnostic embeddings
- Morphological marker integration
- Semantic context encoding

### Phase 3: Sparse Pāṇinian Attention

**Input**: Q, K, V projections + Adjacency Matrix M
**Output**: Attended representations

**Mechanism**:
- Hardware-level routing using M as fetch-mask
- Selective computation of attention scores
- Massive FLOP reduction through sparsity

### Phase 4: Semantic Maturation

**Input**: Attention outputs
**Output**: Refined semantic representations

**Process**:
- Feed-forward network processing
- Higher-order semantic logic
- World knowledge integration

### Phase 5: Grammar-Constrained Decoding

**Input**: Final hidden states
**Output**: Grammatically valid token predictions

**Features**:
- Dynamic vocabulary restriction
- Morphological state tracking
- Hallucination prevention

## Mathematical Foundations

### Adjacency Matrix Generation

The symbolic engine produces a sparse matrix $M$ where:

- $M_{i,j} = 0$ if token $i$ can grammatically govern token $j$
- $M_{i,j} = -\infty$ if the relationship violates Pāṇinian rules

### Attention Computation

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\text{Sparse}\left(\frac{QK^T}{\sqrt{d_k}}\right) + M\right)V$$

Where $\text{Sparse}(\cdot)$ indicates selective computation based on $M$.

## Advantages Over Traditional Transformers

| Aspect | Traditional Transformer | Panini-LM |
|--------|------------------------|-----------|
| Syntax Handling | Statistical approximation | Deterministic rules |
| Attention Complexity | $O(N^2)$ | $O(N \cdot k)$ |
| Positional Encoding | Required | Optional |
| Data Efficiency | High requirements | Significantly reduced |
| Grammatical Accuracy | Learned | Guaranteed |

## Implementation Considerations

- **Hardware Optimization**: Custom GPU kernels for sparse attention
- **Memory Management**: Efficient handling of large adjacency matrices
- **Scalability**: Linear complexity scaling with sequence length
- **Modularity**: Independent development of symbolic and neural components