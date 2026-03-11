# Phase 5 — Grammar-Constrained Decoding

> Guarantee 100% grammatical correctness at generation time.

---

## Overview

Phase 5 is the final output phase. It applies **hard grammar constraints** during autoregressive decoding to ensure that:

1. Every generated token is morphologically well-formed
2. Token sequences maintain grammatical coherence
3. **100% grammatical correctness is guaranteed** (not probabilistic)

**Key insight**: By reusing the Phase 1 morphological engine at decode time, we can compute which tokens are legally continuable from the current morphological state.

---

## Input/Output Contract

### Input

- **hidden_states**: From Phase 4, shape `(batch, seq, d_model)`
- **morphological_state**: Current grammatical context tracking

### Output

- **generated_tokens**: `List[MorphToken]` — guaranteed grammatical
- **statistics**: Generation metadata (tokens rejected, constraint hits)

---

## Dependencies

- **Input**: Phase 4 output (hidden states)
- **Internal**: Phase 1 engine (reused for constraint generation)
- **External**: Vocabulary with morphological tags

---

## Implementation Details

### Morphological State Tracking

```python
@dataclass
class MorphologicalState:
    """
    Tracks grammatical context during generation.
    
    Used to determine which tokens are legal continuations.
    """
    # Active incomplete kāraka relations
    open_relations: List[str]  # e.g., ["kartā-pending", "karma-pending"]
    
    # Number agreement context
    vacana_ctx: Optional[int]  # 1=sg, 2=dual, 3=pl
    
    # Expected inflection patterns
    expected_vibhakti: Optional[List[int]]  # Legal cases
    
    # Partial compound state
    in_compound: bool
    compound_members: List[str]
```

### Grammar Mask Generation

```python
def compute_grammar_mask(
    state: MorphologicalState,
    vocab: MorphVocabulary
) -> torch.Tensor:
    """
    Generate mask over vocabulary based on current grammatical state.
    
    Returns:
        mask: (vocab_size,) tensor where:
            - 0.0 = token is legal
            - -inf = token is illegal
    """
    mask = torch.full((vocab.size,), float('-inf'))
    
    for idx, token_info in enumerate(vocab.token_list):
        if is_legal_continuation(state, token_info):
            mask[idx] = 0.0
    
    return mask
```

### Legality Checking

```python
def is_legal_continuation(
    state: MorphologicalState,
    token_info: TokenInfo
) -> bool:
    """
    Check if token is a legal continuation from current state.
    """
    # Rule 1: Number agreement
    if state.vacana_ctx is not None:
        if token_info.vacana is not None:
            if token_info.vacana != state.vacana_ctx:
                return False
    
    # Rule 2: Vibhakti constraints
    if state.expected_vibhakti:
        if token_info.vibhakti is not None:
            if token_info.vibhakti not in state.expected_vibhakti:
                return False
    
    # Rule 3: Compound formation rules
    if state.in_compound:
        if not can_continue_compound(state.compound_members, token_info):
            return False
    
    # Rule 4: Open relation satisfaction
    # (e.g., if verb expects object, block sentence-end until satisfied)
    if token_info.is_eos and state.open_relations:
        return False
    
    return True
```

### Constrained Decoding Step

```python
def constrained_step(
    hidden: torch.Tensor,          # (batch, d_model) final position
    lm_head: nn.Linear,            # (d_model -> vocab_size)
    state: MorphologicalState,
    vocab: MorphVocabulary,
    temperature: float = 1.0
) -> Tuple[int, MorphologicalState]:
    """
    One step of grammar-constrained autoregressive decoding.
    
    Returns:
        token_id: Selected token (guaranteed grammatical)
        new_state: Updated morphological state
    """
    # Step 1: Compute raw logits
    logits = lm_head(hidden)  # (vocab_size,)
    
    # Step 2: Apply grammar mask
    grammar_mask = compute_grammar_mask(state, vocab)
    masked_logits = logits + grammar_mask  # -inf for illegal tokens
    
    # Step 3: Sample from legal tokens only
    probs = F.softmax(masked_logits / temperature, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()
    
    # Step 4: Update morphological state
    new_state = update_state(state, vocab.token_list[token_id])
    
    return token_id, new_state
```

### State Update

```python
def update_state(
    state: MorphologicalState,
    token: TokenInfo
) -> MorphologicalState:
    """
    Update morphological state after generating a token.
    """
    new_state = copy(state)
    
    # Update number context
    if token.vacana is not None:
        new_state.vacana_ctx = token.vacana
    
    # Close satisfied relations
    if token.type == "tinanta":  # Verb generated
        new_state.open_relations = [
            r for r in state.open_relations
            if not r.startswith("verb-")
        ]
    
    # Open new relations (e.g., transitive verb expects object)
    if token.expects_argument:
        new_state.open_relations.append(f"{token.expects_argument}-pending")
    
    # Update compound state
    if token.starts_compound:
        new_state.in_compound = True
        new_state.compound_members = [token.stem]
    elif state.in_compound:
        if token.ends_compound:
            new_state.in_compound = False
            new_state.compound_members = []
        else:
            new_state.compound_members.append(token.stem)
    
    return new_state
```

### Full Generation Loop

```python
def generate(
    model: PaniniLM,
    prompt_tokens: List[MorphToken],
    max_length: int = 64,
    temperature: float = 1.0
) -> GenerationOutput:
    """
    Generate grammatically constrained text.
    
    Guarantees: Every generated sequence is 100% grammatical.
    """
    # Initialize state from prompt
    state = MorphologicalState(
        open_relations=[],
        vacana_ctx=None,
        expected_vibhakti=None,
        in_compound=False,
        compound_members=[]
    )
    for token in prompt_tokens:
        state = update_state(state, token)
    
    generated = []
    stats = {"rejected": 0, "total": 0}
    
    for _ in range(max_length):
        # Forward pass
        hidden = model.forward(prompt_tokens + generated)
        
        # Constrained decoding
        token_id, state = constrained_step(
            hidden[:, -1],
            model.lm_head,
            state,
            model.vocab,
            temperature
        )
        
        token_info = model.vocab.token_list[token_id]
        generated.append(token_info)
        
        # Track statistics
        mask = compute_grammar_mask(state, model.vocab)
        stats["rejected"] += (mask == float('-inf')).sum().item()
        stats["total"] += model.vocab.size
        
        # Check for EOS
        if token_info.is_eos:
            break
    
    return GenerationOutput(
        tokens=generated,
        text=detokenize(generated),
        stats=stats
    )
```

---

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| Empty legal set | Grammar too restrictive | Relax constraints, fallback to top-k |
| State inconsistency | Bug in update logic | Reset to default state, log error |
| Infinite loop | No EOS reachable | Force EOS after max_length |

### Fallback Strategy

```python
def constrained_step_safe(hidden, lm_head, state, vocab, temperature):
    """Constrained step with fallback for edge cases."""
    grammar_mask = compute_grammar_mask(state, vocab)
    
    # Check if any token is legal
    legal_count = (grammar_mask == 0.0).sum()
    
    if legal_count == 0:
        logging.warning("No legal tokens! Relaxing constraints.")
        # Fallback: allow any token, just log warning
        grammar_mask.fill_(0.0)
    
    # Continue with normal decoding
    ...
```

---

## Test Specifications

### Unit Tests

```python
def test_mask_application():
    """Masked tokens should have zero probability."""
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([0.0, float('-inf'), 0.0, float('-inf')])
    
    masked = logits + mask
    probs = F.softmax(masked, dim=-1)
    
    assert probs[1] < 1e-6  # -inf → 0
    assert probs[3] < 1e-6

def test_number_agreement():
    """Generated subject-verb should agree in number."""
    state = MorphologicalState(vacana_ctx=1)  # Singular
    
    # Singular verb should be legal
    sg_verb = TokenInfo(type="tinanta", vacana=1)
    assert is_legal_continuation(state, sg_verb)
    
    # Plural verb should be illegal
    pl_verb = TokenInfo(type="tinanta", vacana=3)
    assert not is_legal_continuation(state, pl_verb)

def test_state_update():
    """State should correctly track after verb generation."""
    state = MorphologicalState(open_relations=["kartā-pending"])
    verb = TokenInfo(type="tinanta", expects_argument="karma")
    
    new_state = update_state(state, verb)
    
    assert "verb-pending" not in str(new_state.open_relations)
    assert "karma-pending" in new_state.open_relations
```

### Integration Tests

```python
def test_full_generation():
    """Generated text should parse without errors."""
    model = load_model()
    result = generate(model, prompt="rāmaḥ", max_length=10)
    
    # Should parse cleanly through Phase 1
    tokens = ingest_morphology(result.text)
    assert all(t["type"] != "unknown" for t in tokens)

def test_grammaticality_guarantee():
    """100 generations should all be grammatical."""
    model = load_model()
    
    for _ in range(100):
        result = generate(model, prompt="", max_length=20, temperature=1.5)
        # Use external grammar checker
        assert external_grammar_check(result.text) == "valid"
```

---

## Related Documents

- [Data Contracts](../types/data-contracts.md) — `MorphologicalState`, `GrammarMask` definitions
- [Phase 1](phase1-morphology.md) — Morphological engine (reused here)
- [Phase 4](phase4-ffn.md) — Input source
- [Glossary](../GLOSSARY.md) — Kāraka, Vibhakti, Vacana definitions
- [Decoding](../Decoding.md) — Original decoding documentation
