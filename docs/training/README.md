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

### Parameter Breakdown

| Component | Formula | Default Config | Parameters |
|-----------|---------|----------------|------------|
| Token Embedding | `vocab_size × d_model` | 50,000 × 512 | 25,600,000 |
| Type Embedding | `num_types × d_model` | 7 × 512 | 3,584 |
| Per-Layer Attention | `4 × d_model²` | 4 × 512² | 1,048,576 |
| Per-Layer FFN (SwiGLU) | `3 × d_model × d_ff` | 3 × 512 × 768 | 1,179,648 |
| Per-Layer Norms | `2 × d_model × 2` | 2 × 512 × 2 | 2,048 |
| **Per Layer Total** | — | — | **~2.2M** |
| **× num_layers** | — | × 6 | **~13.4M** |
| Output Head | Tied with embedding | — | 0 |

### Configuration Presets

| Config | d_model | heads | layers | FFN expansion | **Total Parameters** |
|--------|---------|-------|--------|---------------|---------------------|
| **Small** | 256 | 4 | 4 | 1.5× | **~15M** |
| **Default** | 512 | 8 | 6 | 1.5× | **~39M** |
| **Base** | 768 | 12 | 8 | 1.5× | **~85M** |
| **Large** | 1024 | 16 | 12 | 2.0× | **~180M** |

### Comparison with Other Models

| Model | Parameters | Notes |
|-------|------------|-------|
| GPT-2 Small | 117M | General English LM |
| BERT-base | 110M | Bidirectional encoder |
| DistilBERT | 66M | Distilled knowledge |
| **Panini-LM Default** | **~39M** | Sanskrit-specific, grammar-assisted |

**Why smaller?** Panini-LM offloads syntax to deterministic Phase 2A rules, uses no positional encoding, and employs reduced FFN expansion (1.5× vs 4×).

---

## Dataset Structure

### Training Data Format

Training data is pre-processed and stored in JSON for efficient loading:

```json
{
  "metadata": {
    "source": "gita.txt",
    "created_at": "2026-03-15T10:30:00",
    "panini_lm_version": "0.1.0",
    "vocab_size": 3841,
    "num_types": 7
  },
  
  "vocab": {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
    "[MASK]": 4,
    "च": 5,
    "न": 6,
    "सः": 7,
    ...
  },
  
  "type_vocab": {
    "subanta": 0,
    "tinanta": 1,
    "avyaya": 2,
    "krdanta": 3,
    "taddhita": 4,
    "samasa": 5,
    "unknown": 6
  },
  
  "samples": [
    {
      "id": "ch01_s0001",
      "chapter": 1,
      "raw_text": "धृतराष्ट्रः उवाच — हे सञ्जय...",
      
      "token_ids": [2, 1130, 17, 7, 234, 89, ..., 3],
      "type_ids": [6, 0, 1, 2, 0, 0, ..., 6],
      "target_ids": [1130, 17, 7, 234, 89, ..., 0],
      
      "adjacency_edges": [
        {"src": 1, "tgt": 2, "link_type": "kartā-kriyā"},
        {"src": 2, "tgt": 1, "link_type": "kartā-kriyā"},
        {"src": 1, "tgt": 1, "link_type": "sva-sambandha"},
        ...
      ],
      
      "seq_len": 16,
      "num_edges": 40,
      "sparsity": 0.156,
      
      "tokens": [
        {"surface": "धृतराष्ट्रः", "stem": "धृतराष्ट्र", "type": "subanta", "attributes": {...}},
        ...
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
| `token_ids` | `List[int]` | Integer IDs including `[BOS]` and `[EOS]` |
| `type_ids` | `List[int]` | Token type encodings (0=subanta, 1=tinanta, etc.) |
| `target_ids` | `List[int]` | Shifted `token_ids` for next-token prediction |
| `adjacency_edges` | `List[{src, tgt, link_type}]` | Sparse grammatical edges |
| `tokens` | `List[MorphToken]` | Full morphological analysis (for debugging) |

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
│                    Training Loop                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  For each batch:                                         │
│                                                          │
│  1. Load pre-computed:                                   │
│     • token_ids, type_ids → Model input                  │
│     • target_ids → Cross-entropy labels                  │
│     • adjacency_edges → Build attention mask M           │
│                                                          │
│  2. Forward pass (Phases 2B → 5):                        │
│     • Embed tokens (no positional encoding)              │
│     • Compute Q, K, V                                    │
│     • Apply sparse attention with mask M                 │
│     • FFN transformation                                 │
│     • Project to logits                                  │
│                                                          │
│  3. Loss computation:                                    │
│     • Primary: CrossEntropy(logits, target_ids)          │
│     • Optional: Attention regularization on M            │
│                                                          │
│  4. Backward + Optimizer step                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Loss Functions

**Primary Loss** — Next-token prediction:
```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # (batch * seq, vocab_size)
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

### PyTorch DataLoader

```python
import json
import torch
from torch.utils.data import Dataset, DataLoader

class PaniniDataset(Dataset):
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
            "token_ids": torch.tensor(s["token_ids"]),
            "type_ids": torch.tensor(s["type_ids"]),
            "target_ids": torch.tensor(s["target_ids"]),
            "edges": s["adjacency_edges"],
            "seq_len": s["seq_len"],
        }

def collate_fn(batch):
    """Pad sequences and build adjacency matrices."""
    max_len = max(b["seq_len"] for b in batch)
    
    token_ids = torch.stack([
        F.pad(b["token_ids"], (0, max_len - b["seq_len"]))
        for b in batch
    ])
    type_ids = torch.stack([
        F.pad(b["type_ids"], (0, max_len - b["seq_len"]))
        for b in batch
    ])
    target_ids = torch.stack([
        F.pad(b["target_ids"], (0, max_len - b["seq_len"]))
        for b in batch
    ])
    
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
        "token_ids": token_ids,
        "type_ids": type_ids,
        "target_ids": target_ids,
        "adjacency_matrix": torch.stack(adj_matrices),
        "padding_mask": token_ids != PAD_ID,
    }
```

### Training Step

```python
def train_step(model, batch, optimizer):
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(
        token_ids=batch["token_ids"],
        type_ids=batch["type_ids"],
        adjacency_matrix=batch["adjacency_matrix"],
    )
    
    # Loss
    loss = F.cross_entropy(
        logits.view(-1, model.vocab_size),
        batch["target_ids"].view(-1),
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
| Phase 2B-5 | Forward pass | Forward pass |
| Decoding | Teacher forcing | Autoregressive + grammar mask |

At inference, the model accepts raw Sanskrit text and runs all phases live:

```python
def generate(model, text: str, max_length: int = 50):
    # Phase 1: Live morphological analysis
    tokens = ingest_morphology(text)
    
    # Phase 2A: Live adjacency computation
    adjacency = build_adjacency_matrix(tokens)
    
    # Encode
    token_ids = tokenizer.encode(tokens)
    
    # Generate autoregressively
    for _ in range(max_length):
        logits = model(token_ids, adjacency.matrix)
        grammar_mask = compute_grammar_mask(state, vocab)
        next_token = sample(logits[-1] + grammar_mask)
        
        if next_token == EOS_ID:
            break
        
        token_ids = torch.cat([token_ids, next_token])
        state = update_state(state, next_token)
    
    return tokenizer.decode(token_ids)
```
