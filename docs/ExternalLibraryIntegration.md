# External Library Integration Guide

This document expands on the external-module integration blueprint and provides concrete pseudocode and integration patterns for each external component.

Phase 1 integration (Morphology)
- Preferred: `vidyut-prakriya` compiled to a Python extension via PyO3.
- Prototype: call `sanskrit-heritage` Python APIs.

Example Python bridge (high-level)
```
# vidyut_py is a PyO3-wrapped Rust module exposing functions: sandhi_resolve, morphological_analyze
from vidyut_py import sandhi_resolve, morphological_analyze

def ingest_morphology_bridge(text: str):
    padas = sandhi_resolve(text)
    tokens = []
    for p in padas:
        tokens.extend(morphological_analyze(p))
    return tokens
```

Notes on PyO3 binding
- Provide a deterministic API surface; avoid returning Rust-specific types — return JSON/string or Python lists/dicts.

Phase 2 integration (Symbolic Parser / Kāraka links)
- Option A: call samsadhani dependency API offline to precompute Kāraka links for training data.
- Option B: implement a Python bridge that consults vidyut metadata and applies deterministic rules.

Example for building `M` using a samsadhani REST API
```
def query_samsadhani(tokens):
    payload = {'tokens': tokens}
    r = requests.post(SAMSADHANI_URL, json=payload)
    return format_to_M(r.json())
```

Phase 3 integration (Sparse attention kernel)
- Implement a Triton kernel for block-sparse compute. Kernel signature should accept: Q, K, V, M_descriptor.
- Provide a CPU/PyTorch fallback using masked dense attention for environments where Triton is unavailable.

High-level deployment plan
1. Provide pure-Python fallbacks for each external dependency so the repo remains runnable for prototyping.
2. Add an integration tests folder `tests/integration/` that validates the deterministic outputs of Phase 1 and Phase 2 on a small corpus.
3. Package the Rust bridge with standard Python `setup.py` or `pyproject.toml` + maturin for reproducible builds.

Security & determinism
- Validate all external outputs against a canonical test-suite. Keep deterministic seeds only for neural components — symbolic components must be fully deterministic.

Sample CI steps (high level)
```
# 1. Build vidyut-prakriya python wheel via maturin
# 2. Install wheel into test env
# 3. Run integration tests that assert known inputs -> known token outputs and matrix M shapes/values
```

Full example: end-to-end pseudocode
```
def end_to_end_step(text):
    tokens = ingest_morphology_bridge(text)          # Phase 1
    M = build_matrix_M(tokens)                       # Phase 2A
    ids = tokens_to_ids(tokens, vocab_map)           # Phase 2B
    X = embedding_layer(ids)                         # embeddings
    Q, K, V = project_qkv(X)
    context = sparse_attention(Q, K, V, M)          # Phase 3 (Triton or fallback)
    matured = semantic_maturation(context)          # Phase 4
    logits = vocab_proj(matured)
    # decoding uses Phase 5 mask per step
    return logits
```

References
- See `external-module.txt` for the original engineering blueprint.
