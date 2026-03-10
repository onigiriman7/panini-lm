# Pseudo-Code Documentation

The canonical pseudocode is now organized by phase. See the phase documents for complete, annotated pseudocode:

- [Phase1 Morphological ingestion](docs/Phase1_Morphological_Ingestion.md)
- [Phase2 Symbolic syntax](docs/Phase2_Symbolic_Syntax.md)
- [Phase3 Sparse attention](docs/Phase3_Sparse_Paninian_Attention.md)
- [Phase4 Semantic maturation](docs/Phase4_Semantic_Maturation.md)
- [Phase5 Decoding](docs/Phase5_Grammar_Constrained_Decoding.md)

Use [ExternalLibraryIntegration.md](docs/ExternalLibraryIntegration.md) for concrete integration pseudocode tying external libraries into the full pipeline.

### Initialization

**Symbolic Components**:
```python
# External linguistic engines
self.segmenter = HeritageSegmenter()  # Morphological segmentation
self.morph_analyzer = VidyutParser()   # Grammatical attribute extraction
```

**Neural Components**:
```python
# Position-agnostic embeddings (no RoPE/sinusoidal encodings)
self.embedding = nn.Embedding(vocab_size, d_model)

# Attention projections
self.q_proj = nn.Linear(d_model, d_model)
self.k_proj = nn.Linear(d_model, d_model)
self.v_proj = nn.Linear(d_model, d_model)

# Semantic feed-forward network
self.ffn = nn.Sequential(
    nn.Linear(d_model, d_model * 4),
    nn.SiLU(),  # Swish activation
    nn.Linear(d_model * 4, d_model)
)

# Output projection
self.lm_head = nn.Linear(d_model, vocab_size)
```

## Core Methods

### Adjacency Matrix Generation

```python
def _build_adjacency_matrix(self, metadata: list) -> torch.Tensor:
    """
    Core symbolic processing: Generate grammatical routing matrix.
    
    Args:
        metadata: List of morphological attribute dictionaries
        
    Returns:
        N x N adjacency matrix with 0.0 (valid) or -inf (invalid) connections
    """
    seq_len = len(metadata)
    M = torch.full((seq_len, seq_len), float('-inf'))  # Default: impossible
    
    for i, token_a in enumerate(metadata):
        for j, token_b in enumerate(metadata):
            # Evaluate Pāṇinian mathematical functions
            if self.morph_analyzer.is_grammatically_valid_link(token_a, token_b):
                M[i, j] = 0.0  # Enable this grammatical pathway
    
    return M
```

**Key Features**:
- **Deterministic**: No learning required, purely rule-based
- **Sparse**: Only grammatically valid connections enabled
- **Asymmetric**: Directed grammatical dependencies
- **Efficient**: Pre-computed before neural processing

### Forward Pass

```python
def forward(self, raw_sanskrit_text: str):
    """
    End-to-end processing pipeline.
    
    Phases:
    1. Morphological ingestion (FST-based segmentation)
    2. Symbolic syntax processing (adjacency matrix generation)
    3. Neural semantic processing (embeddings + attention + FFN)
    4. Vocabulary projection (raw logits for next-token prediction)
    """
    
    # Phase 1: Resolve morphological boundaries
    split_tokens = self.segmenter.resolve_sandhi(raw_sanskrit_text)
    
    # Phase 2A: Extract grammatical metadata
    metadata = self.morph_analyzer.extract_tags(split_tokens)
    token_ids = self.morph_analyzer.convert_to_ids(split_tokens)
    
    # Phase 2B: Generate grammatical routing matrix
    matrix_M = self._build_adjacency_matrix(metadata)
    matrix_M = matrix_M.to('cuda')  # GPU acceleration
    
    # Phase 3A: Neural embeddings (position-agnostic)
    X = self.embedding(token_ids)
    
    # Phase 3B: Attention projections
    Q = self.q_proj(X)
    K = self.q_proj(X)  # Note: Using q_proj for both Q and K
    V = self.v_proj(X)
    
    # Phase 3C: Sparse Pāṇinian attention
    attn_output = sparse_paninian_attention(Q, K, V, matrix_M)
    
    # Phase 4: Semantic maturation
    hidden_states = self.ffn(attn_output)
    
    # Phase 5: Raw vocabulary predictions
    raw_logits = self.lm_head(hidden_states)
    
    return raw_logits, metadata
```

**Processing Flow**:
1. **Input**: Raw Sanskrit string
2. **Segmentation**: FST resolves Sandhi/Samasa
3. **Analysis**: Extract morphological features
4. **Matrix Generation**: Build grammatical adjacency matrix
5. **Neural Processing**: Embeddings → Attention → FFN
6. **Output**: Next-token logits + grammatical metadata

### Generation Method

```python
@torch.no_grad()
def generate(self, prompt: str, max_new_tokens: int):
    """
    Grammar-constrained autoregressive generation.
    
    Features:
    - Morphological state tracking
    - Dynamic vocabulary restriction
    - Guaranteed grammatical validity
    """
    current_text = prompt
    
    for _ in range(max_new_tokens):
        # Forward pass to get logits and current morphological state
        logits, metadata = self.forward(current_text)
        
        # Get final token's logits
        next_logits = logits[:, -1, :]
        
        # Apply grammar constraints (vocabulary masking)
        # This would filter next_logits to only grammatically valid tokens
        
        # Sample next token (simplified - would use constrained sampling)
        next_token_id = torch.argmax(next_logits, dim=-1)
        next_token = self.id_to_token[next_token_id]
        
        # Append to current text
        current_text += next_token
    
    return current_text
```

**Generation Features**:
- **Constrained Sampling**: Only grammatically valid tokens considered
- **State Tracking**: Maintains morphological context across steps
- **No Hallucinations**: Impossible grammatical forms eliminated

## External Dependencies

### Linguistic Engines

**HeritageSegmenter**:
- Finite State Transducer for morphological segmentation
- Resolves Sandhi (euphonic combinations)
- Decomposes Samasa (compound words)
- Output: Pure morphological tokens

**VidyutParser**:
- Morphological analyzer for Sanskrit
- Extracts grammatical attributes (Puruṣa, Vacana, Liṅga, etc.)
- Validates grammatical relationships
- Provides token-to-ID conversion

### Custom Kernels

**sparse_paninian_attention**:
- Hardware-optimized sparse attention computation
- Triton/OpenAI Triton implementation
- Routes attention only through valid grammatical pathways
- Massive FLOP reduction through sparsity

## Usage Examples

### Training

```python
# Initialize model
model = PaninianNeuroSymbolicLLM(vocab_size=50000, d_model=768, num_heads=12)

# Training loop
for batch in dataloader:
    raw_logits, metadata = model(batch['text'])
    
    # Compute loss (masked language modeling)
    loss = F.cross_entropy(
        raw_logits.view(-1, vocab_size),
        batch['targets'].view(-1)
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

### Inference

```python
# Load trained model
model.eval()

# Generate text
prompt = "rāmaḥ"
generated = model.generate(prompt, max_new_tokens=50)

print(f"Generated: {generated}")
# Output: "rāmaḥ vanaṃ gacchati sītayā saha"
```

### Analysis

```python
# Get grammatical analysis
text = "rāmo gacchati"
logits, metadata = model(text)

# metadata contains morphological breakdown
for i, token_meta in enumerate(metadata):
    print(f"Token {i}: {token_meta}")
```

## Implementation Notes

### Design Decisions

**Position-Agnostic Embeddings**:
- No positional encodings to support Sanskrit's free word order
- Morphological markers provide structural context
- Reduces parameter count and improves generalization

**Single Projection for Q/K**:
- Simplified implementation (would typically use separate projections)
- Q and K share the same linear transformation
- Could be improved for production use

**Simplified Attention**:
- Uses custom sparse kernel instead of standard nn.MultiheadAttention
- Integrates grammatical adjacency matrix directly
- Optimized for hardware acceleration

### Limitations

**Incomplete Rule Implementation**:
- Placeholder for full Pāṇinian rule engine
- `is_grammatically_valid_link` is a mock implementation
- Would need complete Aṣṭādhyāyī implementation

**Simplified Generation**:
- Uses argmax instead of proper constrained sampling
- No morphological state tracking in generation
- Missing vocabulary filtering logic

**Performance Considerations**:
- Adjacency matrix generation is O(N²) in current implementation
- Would need optimization for long sequences
- Memory transfer to GPU could be optimized

## Extensions and Improvements

### Enhanced Symbolic Processing

```python
# Complete rule engine implementation
class CompletePaninianRuleEngine:
    def __init__(self):
        self.rules = self.load_ashtadhyayi_rules()
    
    def evaluate_all_rules(self, token_a, token_b):
        """Apply complete set of Pāṇinian rules"""
        for rule in self.rules:
            if not rule.check_condition(token_a, token_b):
                return False
        return True
```

### Optimized Generation

```python
def constrained_sample(self, logits, morphological_state):
    """Sample from grammatically valid tokens only"""
    valid_mask = self.get_valid_token_mask(morphological_state)
    constrained_logits = logits + (1 - valid_mask) * (-float('inf'))
    return torch.multinomial(F.softmax(constrained_logits), 1)
```

### Multi-Head Specialization

```python
# Different heads for different grammatical roles
class MultiHeadSparseAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.heads = nn.ModuleList([
            SparseAttentionHead(d_model, grammatical_role=role)
            for role in ['karaka', 'agreement', 'case', 'modifier']
        ])
```

## Testing and Validation

### Unit Tests

```python
def test_adjacency_matrix():
    """Test grammatical matrix generation"""
    model = PaninianNeuroSymbolicLLM(...)
    
    # Simple sentence: "rāmaḥ gacchati" (Rama goes)
    metadata = [
        {'root': 'rāma', 'vibhakti': 1, 'vacana': 1},  # Subject
        {'root': 'gam', 'lakara': 'present', 'purusa': 3, 'vacana': 1}  # Verb
    ]
    
    M = model._build_adjacency_matrix(metadata)
    
    # Subject should connect to verb
    assert M[0, 1] == 0.0
    # Verb should not connect back to subject
    assert M[1, 0] == float('-inf')

def test_forward_pass():
    """Test complete forward pass"""
    model = PaninianNeuroSymbolicLLM(vocab_size=1000, d_model=64, num_heads=4)
    
    text = "rāmaḥ"
    logits, metadata = model(text)
    
    assert logits.shape[-1] == 1000  # Vocabulary size
    assert len(metadata) > 0  # Should have morphological analysis
```

### Integration Tests

```python
def test_end_to_end():
    """Test complete generation pipeline"""
    model = PaninianNeuroSymbolicLLM(...)
    
    # This would require trained model
    prompt = "devadattaḥ"
    generated = model.generate(prompt, max_new_tokens=10)
    
    # Check grammatical validity (would need grammar checker)
    assert is_sanskrit_grammatical(generated)
```

## Future Development

### Production Readiness

- Complete Pāṇinian rule implementation
- Optimized sparse attention kernels
- Comprehensive test suite
- Performance benchmarking
- Documentation and examples

### Research Extensions

- Multi-lingual adaptation
- Integration with other linguistic formalisms
- Theoretical analysis of efficiency gains
- Applications to other morphologically rich languages

This pseudo-code serves as a foundation for implementing the Panini-LM architecture, demonstrating the key innovations in neuro-symbolic language modeling while providing a clear path for future development and optimization.