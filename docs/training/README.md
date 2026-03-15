# Training Panini-LM

> Dataset structure, training process, and model specifications.

---

## Table of Contents

- [Model Size](#model-size)
- [Dataset Structure](#dataset-structure)
- [Training Process](#training-process)
- [Data Preparation](#data-preparation)
- [Training Loop](#training-loop)

---

## Model Size

### The Token Compression Breakthrough

Panini-LM achieves a **12× reduction in embedding parameters** by using **Factorized Embeddings** instead of standard vocabulary lookup. The model does not embed fully inflected words (*padas*) — it constructs embeddings dynamically from morphological primitives.

| Approach | Vocabulary | Embedding Parameters |
|----------|------------|---------------------|
| **Standard Transformer** | 50,000 surface forms | 50,000 × 512 = **25,600,000** |
| **Panini-LM** | ~4,000 primitives | See breakdown below = **~2,060,000** |

**Result**: Zero OOV errors for any valid Sanskrit inflection + radical parameter reduction.

### Parameter Breakdown

| Component | Formula | Default Config | Parameters |
|-----------|---------|----------------|------------|
| **Factorized Embeddings** | | | |
| └─ Root Embedding | `4000 × d_model` | 4,000 × 512 | **2,048,000** |
| └─ Type Embedding | `7 × d_model` | 7 × 512 | 3,584 |
| └─ Vibhakti Embedding | `9 × d_model` | 9 × 512 | 4,608 |
| └─ Vacana Embedding | `4 × d_model` | 4 × 512 | 2,048 |
| └─ Purusa Embedding | `4 × d_model` | 4 × 512 | 2,048 |
| **Embedding Total** | — | — | **2,060,288** |
| Per-Layer Attention | `4 × d_model²` | 4 × 512² | 1,048,576 |
| Per-Layer FFN (SwiGLU) | `3 × d_model × d_ff` | 3 × 512 × 768 | 1,179,648 |
| Per-Layer Norms | `2 × d_model × 2` | 2 × 512 × 2 | 2,048 |
| **Per Layer Total** | — | — | **~2.2M** |
| **× num_layers** | — | × 6 | **~13.4M** |
| Output Head | `4000 × d_model` (tied or untied) | 4,000 × 512 | 2,048,000 |

### Configuration Presets

| Config | d_model | heads | layers | FFN expansion | **Total Parameters** |
|--------|---------|-------|--------|---------------|---------------------|
| **Small** | 256 | 4 | 4 | 1.5× | **~8M** |
| **Default** | 512 | 8 | 6 | 1.5× | **~18M** |
| **Base** | 768 | 12 | 8 | 1.5× | **~45M** |
| **Large** | 1024 | 16 | 12 | 2.0× | **~100M** |

### Comparison with Other Models

| Model | Parameters | Vocabulary | Notes |
|-------|------------|------------|-------|
| GPT-2 Small | 117M | 50,257 | General English LM |
| BERT-base | 110M | 30,522 | Bidirectional encoder |
| DistilBERT | 66M | 30,522 | Distilled knowledge |
| **Panini-LM Default** | **~18M** | **~4,000** | Sanskrit-specific, grammar-assisted |

**Why is Panini-LM so much smaller?**
1. **Factorized Embeddings**: 12× fewer embedding parameters
2. **Morphological Offloading**: Syntax handled by deterministic Phase 2A rules
3. **No Positional Encoding**: Free word order → position should not bias attention
4. **Reduced FFN Expansion**: 1.5× vs 4× standard — semantics need less capacity

---

## Dataset Structure

### Training Data Format (Factorized Tensors)

Training data uses **factorized tensor representation** instead of flat `token_ids`. Each morphological dimension has its own ID array:

```json
{
  "metadata": {
    "source": "data/gita.txt",
    "created_at": "2026-03-15T09:56:22.956787",
    "panini_lm_version": "0.1.0",
    "description": "Bhagavad Gita training data for Panini-LM (factorized tensors)",
    "factorized_embeddings": true,
    "num_chapters": 7
  },
  
  "vocab": {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
    "[MASK]": 4,
    "हे": 5,
    "सञ्जय": 6,
    "धर्मक्षेत्र": 7,
    "राम": 100,
    "गम्": 101
  },
  
  "type_vocab": {
    "subanta": 0,
    "tinanta": 1,
    "avyaya": 2,
    "krdanta": 3,
    "taddhita": 4,
    "samasa": 5,
    "none": 6
  },
  
  "vibhakti_vocab": {
    "none": 0,
    "prathamā": 1,
    "dvitīyā": 2,
    "tṛtīyā": 3,
    "caturthī": 4,
    "pañcamī": 5,
    "ṣaṣṭhī": 6,
    "saptamī": 7,
    "sambodhana": 8
  },
  
  "samples": [
    {
      "id": "ch01_s0000",
      "chapter": 1,
      "raw_text": "हे सञ्जय, धर्मक्षेत्रे कुरुक्षेत्रे समवेताः युयुत्सवः मामकाः पाण्डवाः च किम् अकुर्वन्।",
      
      "root_ids": [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3],
      "type_ids": [6, 2, 0, 0, 0, 0, 1, 0, 0, 2, 2, 0, 6],
      "vibhakti_ids": [0, 0, 1, 7, 7, 1, 0, 1, 1, 0, 0, 0, 0],
      "vacana_ids": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
      "purusa_ids": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
      
      "target_root_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3, 0],
      
      "adjacency_edges": [
        {"src": 1, "tgt": 6, "link_type": "kartā-kriyā"},
        {"src": 5, "tgt": 6, "link_type": "kartā-kriyā"},
        {"src": 0, "tgt": 1, "link_type": "adjacent"},
        {"src": 0, "tgt": 0, "link_type": "self"}
      ],
      
      "seq_len": 13,
      "num_edges": 41,
      "sparsity": 0.2426,
      
      "tokens": [
        {"surface": "हे", "stem": "हे", "type": "avyaya", "attributes": {}},
        {"surface": "सञ्जय", "stem": "सञ्जय", "type": "subanta", "attributes": {"vibhakti": 1, "vacana": 1}},
        {"surface": "धर्मक्षेत्रे", "stem": "धर्मक्षेत्र", "type": "subanta", "attributes": {"vibhakti": 7, "vacana": 1}}
      ]
    }
  ],
  
  "statistics": {
    "total_samples": 697,
    "total_tokens": 8324,
    "vocab_size": 3378,
    "unk_rate": 0.0,
    "seq_len": {"min": 4, "max": 46, "mean": 13.94},
    "sparsity": {"min": 0.1025, "max": 0.4375, "mean": 0.2388},
    "type_distribution": {"subanta": 5409, "tinanta": 1783, "avyaya": 1132}
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `root_ids` | `List[int]` | Root/stem IDs from ~4000 primitive vocabulary |
| `type_ids` | `List[int]` | Token type: 0=subanta, 1=tiṅanta, 2=avyaya, etc. |
| `vibhakti_ids` | `List[int]` | Case: 0=none, 1=nominative, 2=accusative, 3=instrumental, 4=dative, 5=ablative, 6=genitive, 7=locative, 8=vocative |
| `vacana_ids` | `List[int]` | Number: 0=none, 1=singular, 2=dual, 3=plural |
| `purusa_ids` | `List[int]` | Person: 0=none, 1=3rd (prathama), 2=2nd (madhyama), 3=1st (uttama) |
| `target_root_ids` | `List[int]` | Shifted `root_ids` for next-token prediction |
| `adjacency_edges` | `List[{...}]` | Sparse grammatical edges for attention supervision |
| `tokens` | `List[MorphToken]` | Full morphological analysis (for debugging) |

### Adjacency Edge Types

The `adjacency_edges` encode grammatical relationships for sparse attention:

| Link Type | Description | Example |
|-----------|-------------|---------|
| `kartā-kriyā` | Agent to verb | Nominative noun → verb |
| `karma-kriyā` | Patient to verb | Accusative noun → verb |
| `karaṇa-kriyā` | Instrument to verb | Instrumental noun → verb |
| `sampradāna-kriyā` | Recipient to verb | Dative noun → verb |
| `apādāna-kriyā` | Source to verb | Ablative noun → verb |
| `adhikaraṇa-kriyā` | Location to verb | Locative noun → verb |
| `viśeṣya-viśeṣaṇa` | Adjective to noun | Modifier relationships |
| `adjacent` | Sequential tokens | Local context |
| `self` | Self-attention | Each token to itself |

### Why Factorized Representation?

| Feature | Flat `token_ids` | Factorized Tensors |
|---------|------------------|-------------------|
| OOV handling | Unknown inflection → `[UNK]` | **Zero OOV** — compose from primitives |
| Vocab size | 50,000+ | **~4,000** |
| Embedding params | 25.6M | **2.06M** |
| Morphological structure | Lost in ID lookup | **Preserved structurally** |

### Why Pre-process?

| Approach | Training Speed | File Size |
|----------|----------------|-----------|
| **Raw text** (on-the-fly Phase 1) | Slow (~50ms/sentence/epoch) | Small |
| **Pre-processed** (current) | Fast (skip Phase 1/2A) | Larger |

Pre-processing runs Phase 1 and 2A once; training reuses the results every epoch.

---

## Training Process

### Overview

```
┌─────────────────────────────────────────────────────────┐
│           Training Loop (Factorized Embeddings)          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  For each batch:                                         │
│                                                          │
│  1. Load pre-computed FACTORIZED tensors:                │
│     • root_ids, type_ids, vibhakti_ids,                  │
│       vacana_ids, purusa_ids → Model input               │
│     • target_root_ids → Cross-entropy labels             │
│     • adjacency_edges → Build attention mask M           │
│                                                          │
│  2. Forward pass (Phases 2B → 5):                        │
│     • FACTORIZED EMBEDDING: Sum of 5 primitive vectors   │
│       E(word) = E(root) + E(type) + E(vibhakti) +        │
│                 E(vacana) + E(purusa)                    │
│     • Compute Q, K, V                                    │
│     • Apply sparse attention with mask M                 │
│     • FFN transformation                                 │
│     • Project to logits (~4000 vocabulary)               │
│                                                          │
│  3. Loss computation:                                    │
│     • Primary: CrossEntropy(logits, target_root_ids)     │
│     • Optional: Attention regularization on M            │
│                                                          │
│  4. Backward + Optimizer step                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Loss Functions

**Primary Loss** — Next-root prediction (vocab ~4000):
```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # (batch * seq, ~4000)
    target_ids.view(-1),           # (batch * seq,)
    ignore_index=PAD_ID
)
```

**Optional Auxiliary Loss** — Attention supervision:
```python
# Encourage attention weights to match grammatical edges
attention_loss = kl_div(attention_weights, adjacency_soft_target)
total_loss = loss + 0.1 * attention_loss
```

### Training Configuration

```python
training_config = {
    # Optimizer
    "optimizer": "AdamW",
    "lr": 1e-4,
    "weight_decay": 0.01,
    "betas": (0.9, 0.95),
    
    # Schedule
    "warmup_steps": 1000,
    "total_steps": 50000,
    "lr_schedule": "cosine",
    
    # Batching
    "batch_size": 32,
    "gradient_accumulation": 4,
    "max_seq_len": 128,
    
    # Regularization
    "dropout": 0.1,
    "label_smoothing": 0.1,
}
```

---

## Data Preparation

### Using GitaTrainingBuilder

The `gita_training_builder.py` script converts raw Sanskrit text into factorized tensor training data.

```python
from scripts.gita_training_builder import GitaTrainingBuilder

# Initialize builder
builder = GitaTrainingBuilder()

# Build dataset from gita.txt
dataset = builder.build_dataset("data/gita.txt")

# Save for training
builder.save_dataset(dataset, "data/gita_training.json")
```

### CLI Usage

```bash
# Generate full training dataset from gita.txt
python scripts/gita_training_builder.py data/gita.txt -o data/gita_training.json
```

### Generated Dataset Statistics

The Bhagavad Gita training data (`data/gita_training.json`) contains:

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Samples** | 697 | Sentences across 6+ chapters |
| **Total Tokens** | 8,324 | Content tokens (excluding BOS/EOS) |
| **Vocabulary Size** | 3,378 | Zero OOV by design |
| **Sequence Length** | 4–46 | Min–Max (mean: 13.94) |
| **Sparsity** | 0.10–0.44 | Adjacency matrix density (mean: 0.24) |

**Token Type Distribution**:
- `subanta` (nominals): 5,409 (65%)
- `tinanta` (verbs): 1,783 (21%)
- `avyaya` (indeclinables): 1,132 (14%)

**Chapter Breakdown**:

| Chapter | Name | Sentences | Tokens |
|---------|------|-----------|--------|
| 1 | अर्जुनविषादयोग | 24 | 536 |
| 2 | साङ्ख्ययोग | 71 | 938 |
| 3 | कर्मयोग | 42 | 537 |
| 4 | ज्ञानकर्मसंन्यासयोग | 42 | 495 |
| 5 | कर्मसंन्यासयोग | 29 | 352 |
| 6 | ध्यानयोग | 47 | 549 |

### Dataset Files

| File | Samples | Purpose |
|------|---------|---------|
| `data/gita_training.json` | 697 | Full training dataset (factorized tensors) |
| `data/gita.txt` | — | Raw Sanskrit source text |

---

## Training Loop

### PyTorch DataLoader (Factorized Tensors)

```python
import json
import torch
from torch.utils.data import Dataset, DataLoader

class PaniniDataset(Dataset):
    """Dataset for Panini-LM with factorized token representation."""
    
    def __init__(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.samples = data["samples"]
        self.vocab = data["vocab"]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            # Factorized input tensors (5 parallel ID arrays)
            "root_ids": torch.tensor(s["root_ids"]),
            "type_ids": torch.tensor(s["type_ids"]),
            "vibhakti_ids": torch.tensor(s["vibhakti_ids"]),
            "vacana_ids": torch.tensor(s["vacana_ids"]),
            "purusa_ids": torch.tensor(s["purusa_ids"]),
            # Target for next-token prediction
            "target_root_ids": torch.tensor(s["target_root_ids"]),
            # Structural data
            "edges": s["adjacency_edges"],
            "seq_len": s["seq_len"],
        }

def collate_fn(batch):
    """Pad all factorized tensors and build adjacency matrices."""
    max_len = max(b["seq_len"] for b in batch)
    
    def stack_and_pad(key):
        return torch.stack([
            F.pad(b[key], (0, max_len - b["seq_len"]))
            for b in batch
        ])
    
    # Pad all factorized input tensors
    root_ids = stack_and_pad("root_ids")
    type_ids = stack_and_pad("type_ids")
    vibhakti_ids = stack_and_pad("vibhakti_ids")
    vacana_ids = stack_and_pad("vacana_ids")
    purusa_ids = stack_and_pad("purusa_ids")
    target_root_ids = stack_and_pad("target_root_ids")
    
    # Build attention masks from edges
    adj_matrices = []
    for b in batch:
        M = torch.full((max_len, max_len), float('-inf'))
        for edge in b["edges"]:
            M[edge["src"], edge["tgt"]] = 0.0
        # Pad positions can attend to themselves
        for i in range(b["seq_len"], max_len):
            M[i, i] = 0.0
        adj_matrices.append(M)
    
    return {
        # Factorized input batch (FactorizedTokenBatch)
        "root_ids": root_ids,
        "type_ids": type_ids,
        "vibhakti_ids": vibhakti_ids,
        "vacana_ids": vacana_ids,
        "purusa_ids": purusa_ids,
        # Targets and masks
        "target_root_ids": target_root_ids,
        "adjacency_matrix": torch.stack(adj_matrices),
        "padding_mask": root_ids != PAD_ID,
    }
```

### Training Step

```python
def train_step(model, batch, optimizer):
    optimizer.zero_grad()
    
    # Forward pass with factorized inputs
    logits = model(
        root_ids=batch["root_ids"],
        type_ids=batch["type_ids"],
        vibhakti_ids=batch["vibhakti_ids"],
        vacana_ids=batch["vacana_ids"],
        purusa_ids=batch["purusa_ids"],
        adjacency_matrix=batch["adjacency_matrix"],
    )
    
    # Loss — predict the next ROOT (semantic content)
    # Note: vocab_size is ~4000, not 50,000
    loss = F.cross_entropy(
        logits.view(-1, model.vocab_size),  # (batch * seq, 4000)
        batch["target_root_ids"].view(-1),  # (batch * seq,)
        ignore_index=PAD_ID,
    )
    
    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()
```

---

## Inference vs Training

| Aspect | Training | Inference |
|--------|----------|-----------|
| Phase 1 | Pre-computed | Run live on input |
| Phase 2A | Pre-computed → `adjacency_edges` | Run live on tokens |
| Factorization | Pre-computed → 5 ID tensors | Run live → 5 ID tensors |
| Phase 2B-5 | Forward pass | Forward pass |
| Decoding | Teacher forcing | Autoregressive + grammar mask |

At inference, the model accepts raw Sanskrit text and runs all phases live:

```python
def generate(model, text: str, max_length: int = 50):
    # Phase 1: Live morphological analysis
    tokens = ingest_morphology(text)
    
    # Phase 2A: Live adjacency computation
    adjacency = build_adjacency_matrix(tokens)
    
    # Encode into factorized tensors
    factorized = tokenizer.encode_factorized(tokens)
    # Returns: root_ids, type_ids, vibhakti_ids, vacana_ids, purusa_ids
    
    # Generate autoregressively
    for _ in range(max_length):
        logits = model(
            root_ids=factorized["root_ids"],
            type_ids=factorized["type_ids"],
            vibhakti_ids=factorized["vibhakti_ids"],
            vacana_ids=factorized["vacana_ids"],
            purusa_ids=factorized["purusa_ids"],
            adjacency_matrix=adjacency.matrix,
        )
        
        # Grammar-constrained decoding
        grammar_mask = compute_grammar_mask(state, vocab)
        next_root = sample(logits[-1] + grammar_mask)
        
        if next_root == EOS_ID:
            break
        
        # Update factorized tensors for next step
        factorized = extend_factorized(factorized, next_root, state)
        state = update_state(state, next_root)
    
    return tokenizer.decode_factorized(factorized)
```

---

## Morphological Analysis

### Current Implementation

The `gita_training_builder.py` uses a **heuristic-based analyzer** for morphological analysis. This provides reasonable results for common patterns but can be enhanced with proper morphological backends.

**Heuristic features**:
- Common avyaya (indeclinable) detection: च, वा, न, तु, हि, एव, अपि, etc.
- Verb ending recognition: -ति, -ते, -न्ति, -तु, -ष्यति, etc.
- Case ending patterns: -ः (nom), -म् (acc), -ेन (inst), -ाय (dat), etc.

### Improving Analysis with Vidyut

For production-quality training data, integrate [vidyut-prakriya](../integration/vidyut.md):

```python
# Enhanced analyzer using vidyut
from vidyut_py import Vyakarana

def analyze_token_vidyut(surface: str) -> MorphToken:
    """Analyze token using vidyut-prakriya for accurate morphology."""
    v = Vyakarana()
    
    # Get all possible analyses
    analyses = v.analyze(surface)
    
    if not analyses:
        return fallback_heuristic(surface)
    
    # Select best analysis (typically first)
    analysis = analyses[0]
    
    return {
        "surface": surface,
        "stem": analysis.pratipadika or analysis.dhatu,
        "type": map_vidyut_type(analysis.pada_type),
        "attributes": extract_vidyut_attributes(analysis)
    }
```

### Morphological Backend Comparison

| Backend | Accuracy | Speed | Coverage | Notes |
|---------|----------|-------|----------|-------|
| **Heuristic** | ~70% | Fast | Basic patterns | Current default |
| **vidyut-prakriya** | ~95% | Medium | Comprehensive | Recommended |
| **sanskrit-heritage** | ~90% | Slow | Classical texts | Alternative |
| **samsadhani** | ~92% | Medium | Kāraka analysis | Adds semantic roles |

For training data preparation, accuracy is critical. Consider using vidyut-prakriya for the final training dataset while using the heuristic analyzer for rapid prototyping.

---

## File Locations

| File | Path | Description |
|------|------|-------------|
| Training data | `data/gita_training.json` | Generated training dataset |
| Source text | `data/gita.txt` | Raw Bhagavad Gita prose |
| Builder script | `scripts/gita_training_builder.py` | Training data generator |
| Legacy builder | `scripts/training_data_builder.py` | Original builder (requires panini_lm) |
| Gita parser | `scripts/gita_parser.py` | Chapter/sentence extraction |

