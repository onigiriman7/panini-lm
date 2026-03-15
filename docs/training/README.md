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
    "source": "gita.txt",
    "created_at": "2026-03-15T10:30:00",
    "panini_lm_version": "0.1.0",
    "vocab_size": 4000,
    "num_types": 7,
    "num_vibhakti": 9,
    "num_vacana": 4,
    "num_purusa": 4
  },
  
  "vocab": {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
    "[MASK]": 4,
    "रम्": 5,
    "गम्": 6,
    "भू": 7,
    "राम": 100,
    "गृह": 101,
    ...
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
  
  "samples": [
    {
      "id": "ch01_s0001",
      "chapter": 1,
      "raw_text": "रामो गृहं गच्छति",
      
      "root_ids": [2, 100, 101, 6, 3],
      "type_ids": [6, 0, 0, 1, 6],
      "vibhakti_ids": [0, 1, 2, 0, 0],
      "vacana_ids": [0, 1, 1, 1, 0],
      "purusa_ids": [0, 0, 0, 1, 0],
      
      "target_root_ids": [100, 101, 6, 3, 0],
      
      "adjacency_edges": [
        {"src": 1, "tgt": 3, "link_type": "kartā-kriyā"},
        {"src": 2, "tgt": 3, "link_type": "karma-kriyā"},
        ...
      ],
      
      "seq_len": 5,
      "num_edges": 6,
      "sparsity": 0.24,
      
      "tokens": [
        {"surface": "रामः", "stem": "राम", "type": "subanta", "attributes": {"vibhakti": 1, "vacana": 1, "linga": "m"}},
        {"surface": "गृहम्", "stem": "गृह", "type": "subanta", "attributes": {"vibhakti": 2, "vacana": 1, "linga": "n"}},
        {"surface": "गच्छति", "stem": "गम्", "type": "tinanta", "attributes": {"purusa": 1, "vacana": 1, "lakara": "lat"}}
      ]
    },
    ...
  ],
  
  "statistics": {
    "total_samples": 1242,
    "total_tokens": 11523,
    "unk_rate": 0.0,
    "seq_len": {"min": 3, "max": 34, "mean": 9.28},
    "sparsity": {"min": 0.08, "max": 0.28, "mean": 0.23}
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `root_ids` | `List[int]` | Root/stem IDs from ~4000 primitive vocabulary |
| `type_ids` | `List[int]` | Token type: 0=subanta, 1=tiṅanta, 2=avyaya, etc. |
| `vibhakti_ids` | `List[int]` | Case: 0=none, 1-7=cases, 8=vocative |
| `vacana_ids` | `List[int]` | Number: 0=none, 1=sing, 2=dual, 3=plural |
| `purusa_ids` | `List[int]` | Person: 0=none, 1=3rd, 2=2nd, 3=1st |
| `target_root_ids` | `List[int]` | Shifted `root_ids` for next-token prediction |
| `adjacency_edges` | `List[{...}]` | Sparse grammatical edges for attention mask |
| `tokens` | `List[MorphToken]` | Full morphological analysis (for debugging) |

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

### Using TrainingDataBuilder

```python
from scripts.training_data_builder import TrainingDataBuilder

# Initialize builder
builder = TrainingDataBuilder()

# Process Gita text
dataset = builder.process_file(
    "gita.txt",
    max_sentences=None,  # All sentences
    chapters=None,       # All chapters
)

# Save for training
builder.save_dataset(dataset, "tests/data/gita_training.json")
```

### CLI Usage

```bash
# Full dataset
python scripts/training_data_builder.py gita.txt -o training.json

# Subset (chapters 1-3, max 10 sentences each)
python scripts/training_data_builder.py gita.txt -o subset.json -c 1 2 3 -n 10

# Verbose output
python scripts/training_data_builder.py gita.txt -o training.json -v
```

### Dataset Files

| File | Samples | Purpose |
|------|---------|---------|
| `gita_training.json` | 1,242 | Full training dataset |
| `gita_samples.json` | 10 | Quick testing/debugging |

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
