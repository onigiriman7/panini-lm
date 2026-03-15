# External Library Integration

> Integration guides for external libraries used in Panini-LM.

---

## Overview

Panini-LM relies on several external libraries for morphological analysis, symbolic processing, and GPU-optimized attention.

### Library Summary

| Library | Phase | Purpose | Required |
|---------|-------|---------|----------|
| [vidyut-prakriya](vidyut.md) | 1 | Production morphological analysis + factorization | Yes (or fallback) |
| [sanskrit-heritage](sanskrit-heritage.md) | 1 | Fallback morphological analysis | Fallback only |
| [samsadhani](samsadhani.md) | 2A | Kāraka relationship API | Optional |
| [Triton](triton.md) | 3 | Block-sparse attention kernel | Optional (GPU) |
| PyTorch | 2B-5 | Neural components (factorized embeddings) | Yes |

---

## Quick Start

### Minimal Setup (CPU/Prototyping)

```bash
pip install torch sanskrit-heritage
```

### Production Setup (GPU)

```bash
# Install vidyut-prakriya via maturin (Rust → Python)
pip install maturin
cd vidyut-prakriya && maturin develop --release

# Install Triton
pip install triton

# Install Panini-LM
pip install -e .
```

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       PANINI-LM CORE                            │
│                 (Factorized Embedding Architecture)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Phase 1    │    │  Phase 2A   │    │  Phase 3    │         │
│  │  Morphology │    │  Symbolic   │    │  Attention  │         │
│  │  + FACTOR-  │    └──────┬──────┘    └──────┬──────┘         │
│  │  IZATION    │           │                  │                 │
│  └──────┬──────┘           │                  │                 │
│         │                  │                  │                 │
├─────────┼──────────────────┼──────────────────┼─────────────────┤
│         ▼                  ▼                  ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ VIDYUT      │    │ SAMSADHANI  │    │ TRITON      │         │
│  │ (Rust/PyO3) │    │ (REST API)  │    │ (CUDA)      │         │
│  ├─────────────┤    └─────────────┘    ├─────────────┤         │
│  │ HERITAGE    │                       │ PYTORCH     │         │
│  │ (Python)    │                       │ (fallback)  │         │
│  └─────────────┘                       └─────────────┘         │
│    ▲ fallback                            ▲ fallback            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### What Each Library Provides

| Library | Output for Panini-LM |
|---------|---------------------|
| vidyut/heritage | MorphTokens + **FactorizedTokenBatch** (5 parallel ID tensors) |
| samsadhani | Kāraka relationships for training data |
| Triton | O(N·k) sparse attention with true FLOP savings |
| PyTorch | Factorized embedding layers, Q/K/V projection |

---

## Fallback Strategy

Each external dependency has a fallback mechanism:

| Primary | Fallback | Trigger |
|---------|----------|---------|
| vidyut-prakriya | sanskrit-heritage | `ImportError` |
| Triton kernel | PyTorch masked attention | `ImportError` or CPU |
| samsadhani API | Local rule engine | Network error |

### Fallback Implementation Pattern

```python
def get_backend(backend_type: str):
    """Get appropriate backend with fallback."""
    if backend_type == "morphology":
        try:
            from vidyut_py import analyzer
            return analyzer
        except ImportError:
            from sanskrit_heritage import analyzer
            return analyzer
    
    elif backend_type == "attention":
        if torch.cuda.is_available():
            try:
                from panini_lm.kernels import triton_attention
                return triton_attention
            except ImportError:
                pass
        from panini_lm.attention import pytorch_attention
        return pytorch_attention
```

---

## Library Details

### [vidyut-prakriya](vidyut.md)

High-performance Sanskrit morphological analyzer in Rust.

- **Language**: Rust with PyO3 Python bindings
- **Installation**: `maturin develop --release`
- **Performance**: ~10x faster than Python alternatives
- **Features**: Sandhi resolution, morphological analysis, attribute extraction

### [sanskrit-heritage](sanskrit-heritage.md)

Pure Python Sanskrit analysis toolkit.

- **Language**: Python
- **Installation**: `pip install sanskrit-heritage`
- **Use case**: Prototyping, fallback, easier debugging
- **Features**: Sandhi splitting, dictionary lookup

### [samsadhani](samsadhani.md)

University of Hyderabad's computational linguistics platform.

- **Interface**: REST API
- **Use case**: Pre-computing Kāraka links for training data
- **Features**: Dependency parsing, Kāraka identification

### [Triton](triton.md)

OpenAI's GPU kernel language.

- **Language**: Python DSL → CUDA
- **Installation**: `pip install triton`
- **Use case**: Block-sparse attention with true FLOP savings
- **Features**: Custom kernels, automatic optimization

---

## Configuration

### Environment Variables

```bash
# Morphology backend selection
export PANINI_MORPH_BACKEND="vidyut"  # or "heritage"

# Attention backend selection
export PANINI_ATTENTION_BACKEND="triton"  # or "pytorch"

# samsadhani API (optional)
export SAMSADHANI_URL="https://sanskrit.uohyd.ac.in/cgi-bin/scl/api"
```

### Programmatic Configuration

```python
from panini_lm import Config

config = Config(
    morph_backend="vidyut",
    morph_fallback="heritage",
    attention_backend="triton",
    attention_fallback="pytorch",
    samsadhani_url=None,  # Disable external API
)
```

---

## Testing Integration

```bash
# Test morphology backends
pytest tests/integration/test_morphology.py -v

# Test attention backends
pytest tests/integration/test_attention.py -v

# Test with specific backend
PANINI_MORPH_BACKEND=heritage pytest tests/test_phase1.py -v
```

---

## Related Documentation

- [Data Contracts](../types/data-contracts.md) — Interface types
- [Test Specifications](../testing/test-specifications.md) — Integration tests
- [Phase Overview](../phases/README.md) — Where libraries are used
