# Phase 2 — Symbolic Engine (Syntax Track)

This document covers both Phase 2A (deterministic matrix generation) and Phase 2B (semantic token mapping).

Phase 2A — Matrix M generation
- Purpose: evaluate grammatical links (Kāraka relationships) deterministically from the morphological token list.
- Output: PyTorch tensor `M` of shape (N, N) where allowed edges are 0.0 and impossible edges are -inf.

Pseudocode (Matrix generation)
```
def build_matrix_M(tokens: List[Dict]) -> torch.Tensor:
    N = len(tokens)
    M = torch.full((N, N), float('-inf'), dtype=torch.float32)
    for i, t_i in enumerate(tokens):
        for j, t_j in enumerate(tokens):
            if is_grammatically_valid(t_i, t_j):
                M[i, j] = 0.0
    return M

def is_grammatically_valid(head, dependent) -> bool:
    # deterministic rules based on attributes (puruṣa, vacana, vibhakti, lakāra, etc.)
    # implement Aṣṭādhyāyī-derived constraints here or call samsadhani bridge
    return rule_engine.check(head, dependent)
```

Phase 2B — Semantic tokenizer and embeddings
- Map `stem` and `type` to integer token ids. No positional encodings are added.

Pseudocode (token mapping -> embeddings)
```
def tokens_to_ids(tokens, vocab_map) -> torch.LongTensor:
    ids = [vocab_map.get((t['stem'], t['type']), vocab_map['<unk>']) for t in tokens]
    return torch.tensor(ids, dtype=torch.long)

def embed_tokens(ids, embedding_layer) -> torch.Tensor:
    # returns (seq_len, d_model)
    return embedding_layer(ids)
```

Notes
- The symbolic tensor `M` is computed entirely from Phase 1 attributes; it is deterministic and detached from model gradients.
