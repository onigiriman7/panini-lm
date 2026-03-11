# samsadhani Integration

> University of Hyderabad Sanskrit computational linguistics API.

---

## Overview

**samsadhani** is a comprehensive Sanskrit analysis platform providing dependency parsing and Kāraka identification via REST API.

- **URL**: https://sanskrit.uohyd.ac.in/
- **Use case**: Pre-computing Kāraka relationships for training data
- **Phase**: 2A (Symbolic Engine)

---

## API Usage

### Endpoint

```
POST https://sanskrit.uohyd.ac.in/cgi-bin/scl/samsaadhanii/pdartha_nirnaya.cgi
```

### Request

```python
import requests

def query_samsadhani(sentence: str) -> dict:
    """Query samsadhani for Kāraka analysis."""
    url = "https://sanskrit.uohyd.ac.in/cgi-bin/scl/samsaadhanii/pdartha_nirnaya.cgi"
    
    response = requests.post(url, data={
        "text": sentence,
        "encoding": "Unicode",
        "outencoding": "Unicode"
    })
    
    return parse_response(response.text)
```

### Response Format

```json
{
  "sentence": "rāmaḥ gṛham gacchati",
  "analysis": [
    {
      "word": "rāmaḥ",
      "karaka": "kartā",
      "head": "gacchati"
    },
    {
      "word": "gṛham", 
      "karaka": "karma",
      "head": "gacchati"
    },
    {
      "word": "gacchati",
      "karaka": "kriyā",
      "head": null
    }
  ]
}
```

---

## Panini-LM Integration

### Training Data Preparation

Use samsadhani to pre-compute Kāraka links for training corpus:

```python
def precompute_karaka_links(corpus: List[str]) -> List[dict]:
    """Batch process corpus through samsadhani."""
    results = []
    
    for sentence in tqdm(corpus):
        try:
            analysis = query_samsadhani(sentence)
            results.append({
                "sentence": sentence,
                "links": extract_links(analysis)
            })
        except Exception as e:
            logging.warning(f"Failed: {sentence}: {e}")
            results.append({"sentence": sentence, "links": None})
    
    return results

def extract_links(analysis: dict) -> List[tuple]:
    """Convert samsadhani output to edge list."""
    links = []
    words = analysis["analysis"]
    
    for i, word in enumerate(words):
        if word["head"]:
            head_idx = next(
                j for j, w in enumerate(words) 
                if w["word"] == word["head"]
            )
            links.append((i, head_idx, word["karaka"]))
    
    return links
```

### Building Matrix M

```python
def build_M_from_samsadhani(analysis: dict, seq_len: int) -> torch.Tensor:
    """Build adjacency matrix from samsadhani analysis."""
    M = torch.full((seq_len, seq_len), float('-inf'))
    
    links = extract_links(analysis)
    for src, tgt, karaka in links:
        M[src, tgt] = 0.0
        M[tgt, src] = 0.0  # Bidirectional
    
    # Self-attention always allowed
    for i in range(seq_len):
        M[i, i] = 0.0
    
    return M
```

---

## Offline vs Online

| Mode | Use Case | Latency |
|------|----------|---------|
| Offline batch | Training data prep | Acceptable |
| Online query | Runtime inference | Too slow |

**Recommendation**: Use samsadhani offline to prepare training data, then train a local rule engine for runtime.

---

## Fallback

When samsadhani is unavailable, use local rule engine:

```python
def get_karaka_links(tokens: List[MorphToken]) -> List[tuple]:
    """Get Kāraka links with samsadhani fallback."""
    try:
        analysis = query_samsadhani(tokens_to_sentence(tokens))
        return extract_links(analysis)
    except (requests.RequestException, TimeoutError):
        # Fall back to local rules
        return local_karaka_inference(tokens)
```

---

## Related Documentation

- [Phase 2A — Symbolic Engine](../phases/phase2a-symbolic.md)
- [vidyut](vidyut.md) — Alternative metadata source
