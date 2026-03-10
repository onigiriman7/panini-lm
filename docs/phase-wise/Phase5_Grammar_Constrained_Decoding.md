# Phase 5 — Grammar-Constrained Decoding

Goal
- During autoregressive decoding, forbid grammatically illegal tokens by adding a mask to logits.

Pseudocode (decoding step)
```
def constrained_step(hidden_state, morphological_state, decode_engine, vocab_projection):
    raw_logits = vocab_projection(hidden_state)
    legal_mask = decode_engine.allowed_tokens(morphological_state)  # boolean array length vocab_size
    mask_tensor = torch.where(legal_mask, 0.0, float('-inf'))
    constrained_logits = raw_logits + mask_tensor
    probs = torch.softmax(constrained_logits, dim=-1)
    next_token = sample_from_probs(probs)
    morphological_state = update_morph_state(morphological_state, next_token)
    return next_token, morphological_state
```

Integration note
- `decode_engine.allowed_tokens()` should be a deterministic query to the morphological engine (Phase 1) or a cached rule table.
