# Phase 3 — Sparse Pāṇinian Attention

Goal
- Use matrix `M` from Phase 2 as a routing mask to avoid computing dense O(N^2) attention for impossible links.

Design choices
- Preferred: custom Triton block-sparse kernel that skips blocks where `M == -inf`.
- Fallback: use PyTorch's masked attention but accept extra compute (masking after dot-product).

High-level Triton pseudocode
```
# Kernel input: Q, K, V, M
for block_i in seq_blocks:
    for block_j in seq_blocks:
        if M_block[block_i, block_j] == -inf:
            continue  # skip loading K_block and computing dot
        scores = dot(Q_block_i, K_block_j.T) / sqrt(dk)
        attention = softmax(scores + M_block)
        out_block_i += attention @ V_block_j
```

Fallback PyTorch pseudocode
```
def sparse_attention(Q, K, V, M):
    scores = Q @ K.transpose(-2, -1) / math.sqrt(dk)
    scores = scores + M  # M contains -inf where disallowed
    attn = torch.softmax(scores, dim=-1)
    return attn @ V
```

Notes
- The Triton kernel must accept `M` as a sparse adjacency descriptor (block or row-sparse format) to avoid loading K blocks.
