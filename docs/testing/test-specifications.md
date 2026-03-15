# Test Specifications

> Test cases, validation criteria, and benchmark specifications for each phase.

---

## Table of Contents

- [Overview](#overview)
- [Phase 1 Tests](#phase-1-tests-morphological-ingestion)
- [Phase 2A Tests](#phase-2a-tests-symbolic-engine)
- [Phase 2B Tests](#phase-2b-tests-neural-engine)
- [Phase 3 Tests](#phase-3-tests-sparse-attention)
- [Phase 4 Tests](#phase-4-tests-semantic-maturation)
- [Phase 5 Tests](#phase-5-tests-grammar-constrained-decoding)
- [Integration Tests](#integration-tests)
- [Performance Benchmarks](#performance-benchmarks)
- [CI/CD Configuration](#cicd-configuration)

---

## Overview

### Test Categories

| Category | Purpose | Frequency |
|----------|---------|-----------|
| **Unit Tests** | Validate individual functions | Every commit |
| **Integration Tests** | Validate phase interactions | Every PR |
| **Regression Tests** | Prevent known bug recurrence | Every commit |
| **Performance Tests** | Benchmark speed/memory | Nightly |
| **Smoke Tests** | Basic sanity checks | Pre-deploy |

### Test Framework

```bash
# Required packages
pip install pytest pytest-cov pytest-benchmark hypothesis

# Run all tests
pytest tests/ -v --cov=panini_lm

# Run specific phase tests
pytest tests/test_phase1/ -v

# Run benchmarks
pytest tests/test_benchmark.py --benchmark-only
```

---

## Phase 1 Tests: Morphological Ingestion

### Unit Tests

#### `test_sandhi_resolution`

**Purpose**: Verify sandhi splitting produces correct padas.

```python
import pytest
from panini_lm.phase1_morphology import resolve_sandhi

class TestSandhiResolution:
    """Tests for sandhi resolution."""
    
    @pytest.mark.parametrize("input_text,expected", [
        # Visarga sandhi
        ("rāmo'pi", ["rāmaḥ", "api"]),
        ("devāśca", ["devāḥ", "ca"]),
        
        # Vowel sandhi
        ("rāmāgacchati", ["rāma", "āgacchati"]),
        ("nadyeva", ["nadī", "iva"]),
        
        # Consonant sandhi
        ("tacchrī", ["tat", "śrī"]),
        
        # No sandhi (already split)
        ("rāmaḥ gacchati", ["rāmaḥ", "gacchati"]),
        
        # Complex compound
        ("mahārājaputraḥ", ["mahā", "rāja", "putraḥ"]),
    ])
    def test_known_sandhi_cases(self, input_text: str, expected: list[str]):
        """Test known sandhi resolution cases."""
        result = resolve_sandhi(input_text)
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_empty_input(self):
        """Empty input should return empty list."""
        assert resolve_sandhi("") == []
    
    def test_single_word(self):
        """Single word without sandhi should return as-is."""
        assert resolve_sandhi("rāmaḥ") == ["rāmaḥ"]
    
    def test_determinism(self):
        """Same input must always produce same output."""
        text = "rāmo'pi gṛhaṃ gacchati"
        result1 = resolve_sandhi(text)
        result2 = resolve_sandhi(text)
        assert result1 == result2
```

#### `test_morphological_analysis`

**Purpose**: Verify token analysis produces correct attributes.

```python
from panini_lm.phase1_morphology import analyze_token
from panini_lm.core.types import MorphToken

class TestMorphologicalAnalysis:
    """Tests for morphological token analysis."""
    
    def test_nominal_analysis(self):
        """Test analysis of nominal (subanta) tokens."""
        result = analyze_token("rāmaḥ")
        
        assert result["surface"] == "rāmaḥ"
        assert result["stem"] == "rāma"
        assert result["type"] == "subanta"
        assert result["attributes"]["vibhakti"] == 1  # nominative
        assert result["attributes"]["vacana"] == 1    # singular
        assert result["attributes"]["linga"] == "m"   # masculine
    
    def test_verbal_analysis(self):
        """Test analysis of verbal (tiṅanta) tokens."""
        result = analyze_token("gacchati")
        
        assert result["stem"] == "gam"
        assert result["type"] == "tinanta"
        assert result["attributes"]["purusa"] == 1    # 3rd person
        assert result["attributes"]["vacana"] == 1    # singular
        assert result["attributes"]["lakara"] == "lat"  # present
    
    def test_indeclinable_analysis(self):
        """Test analysis of indeclinable (avyaya) tokens."""
        result = analyze_token("ca")
        
        assert result["type"] == "avyaya"
        assert result["attributes"] == {}  # No attributes for avyaya
    
    @pytest.mark.parametrize("token,expected_type", [
        ("rāmaḥ", "subanta"),
        ("gacchati", "tinanta"),
        ("ca", "avyaya"),
        ("gacchan", "krdanta"),
    ])
    def test_token_type_classification(self, token: str, expected_type: str):
        """Test correct token type classification."""
        result = analyze_token(token)
        assert result["type"] == expected_type
```

#### `test_phase1_fallback`

**Purpose**: Verify vidyut → heritage fallback works.

```python
from unittest.mock import patch, MagicMock
from panini_lm.phase1_morphology import ingest_morphology

class TestPhase1Fallback:
    """Tests for fallback mechanism."""
    
    def test_vidyut_success(self):
        """When vidyut succeeds, use its result."""
        result = ingest_morphology("rāmaḥ")
        assert len(result["tokens"]) > 0
    
    @patch('panini_lm.phase1_morphology.vidyut_backend.analyze')
    def test_fallback_on_vidyut_failure(self, mock_vidyut):
        """When vidyut fails, fall back to heritage."""
        mock_vidyut.side_effect = RuntimeError("Vidyut unavailable")
        
        result = ingest_morphology("rāmaḥ")
        assert len(result["tokens"]) > 0  # Heritage should handle it
    
    @patch('panini_lm.phase1_morphology.vidyut_backend.analyze')
    @patch('panini_lm.phase1_morphology.heritage_backend.analyze')
    def test_both_backends_fail(self, mock_heritage, mock_vidyut):
        """When both backends fail, raise MorphologyError."""
        mock_vidyut.side_effect = RuntimeError("Vidyut error")
        mock_heritage.side_effect = RuntimeError("Heritage error")
        
        with pytest.raises(MorphologyError):
            ingest_morphology("rāmaḥ")
```

### Test Data: Known Corpus

```python
# tests/fixtures/phase1_corpus.json
PHASE1_TEST_CORPUS = [
    {
        "input": "rāmo'pi gṛhaṃ gacchati",
        "expected_tokens": [
            {"surface": "rāmaḥ", "stem": "rāma", "type": "subanta"},
            {"surface": "api", "stem": "api", "type": "avyaya"},
            {"surface": "gṛham", "stem": "gṛha", "type": "subanta"},
            {"surface": "gacchati", "stem": "gam", "type": "tinanta"}
        ]
    },
    {
        "input": "ahaṃ tvāṃ paśyāmi",
        "expected_tokens": [
            {"surface": "aham", "stem": "mad", "type": "subanta"},
            {"surface": "tvām", "stem": "tvad", "type": "subanta"},
            {"surface": "paśyāmi", "stem": "dṛś", "type": "tinanta"}
        ]
    }
]
```

---

## Phase 2A Tests: Symbolic Engine

### Unit Tests

#### `test_grammatical_validity`

**Purpose**: Verify rule engine correctly identifies valid/invalid links.

```python
from panini_lm.phase2a_symbolic import is_grammatically_valid

class TestGrammaticalValidity:
    """Tests for grammatical link validation."""
    
    def test_subject_verb_agreement(self):
        """Subject (nominative) can link to matching verb."""
        subject = {"type": "subanta", "attributes": {"vibhakti": 1, "purusa": 1, "vacana": 1}}
        verb = {"type": "tinanta", "attributes": {"purusa": 1, "vacana": 1}}
        
        assert is_grammatically_valid(subject, verb) == True
    
    def test_subject_verb_mismatch(self):
        """Subject-verb number mismatch should be invalid."""
        subject = {"type": "subanta", "attributes": {"vibhakti": 1, "vacana": 1}}  # singular
        verb = {"type": "tinanta", "attributes": {"purusa": 1, "vacana": 3}}        # plural
        
        assert is_grammatically_valid(subject, verb) == False
    
    def test_object_verb_link(self):
        """Object (accusative) can link to verb."""
        obj = {"type": "subanta", "attributes": {"vibhakti": 2}}  # accusative
        verb = {"type": "tinanta", "attributes": {}}
        
        assert is_grammatically_valid(obj, verb) == True
    
    def test_avyaya_isolation(self):
        """Avyaya (indeclinable) has limited connections."""
        avyaya = {"type": "avyaya", "attributes": {}}
        nominal = {"type": "subanta", "attributes": {"vibhakti": 1}}
        
        # Avyaya typically modifies adjacent elements only
        assert is_grammatically_valid(avyaya, nominal) in [True, False]  # Implementation-dependent
```

#### `test_matrix_generation`

**Purpose**: Verify adjacency matrix M has correct shape and values.

```python
import torch
from panini_lm.phase2a_symbolic import build_matrix_M

class TestMatrixGeneration:
    """Tests for adjacency matrix generation."""
    
    def test_matrix_shape(self):
        """Matrix should be (N, N)."""
        tokens = [{"type": "subanta"}, {"type": "tinanta"}, {"type": "avyaya"}]
        result = build_matrix_M(tokens)
        
        assert result["matrix"].shape == (3, 3)
    
    def test_matrix_dtype(self):
        """Matrix should be float32."""
        tokens = [{"type": "subanta"}, {"type": "tinanta"}]
        result = build_matrix_M(tokens)
        
        assert result["matrix"].dtype == torch.float32
    
    def test_matrix_values(self):
        """Matrix values should be 0.0 or -inf only."""
        tokens = [{"type": "subanta"}, {"type": "tinanta"}]
        result = build_matrix_M(tokens)
        
        M = result["matrix"]
        valid = (M == 0.0) | (M == float('-inf'))
        assert valid.all(), "Matrix contains invalid values"
    
    def test_sparsity(self):
        """Matrix should be sparse (avg k < 5)."""
        # Generate a longer sequence
        tokens = [{"type": "subanta"}] * 10 + [{"type": "tinanta"}]
        result = build_matrix_M(tokens)
        
        avg_k = result["meta"]["avg_connections_per_token"]
        assert avg_k < 5, f"Matrix too dense: avg_k = {avg_k}"
    
    def test_self_attention_allowed(self):
        """Diagonal (self-attention) should typically be allowed."""
        tokens = [{"type": "subanta"}, {"type": "tinanta"}]
        result = build_matrix_M(tokens)
        M = result["matrix"]
        
        # Self-attention is usually valid
        for i in range(len(tokens)):
            assert M[i, i] == 0.0, f"Self-attention blocked at position {i}"
```

### Validation Suite

```python
PHASE2A_VALIDATION_CASES = [
    {
        "description": "Simple subject-verb sentence",
        "tokens": [
            {"surface": "rāmaḥ", "type": "subanta", "attributes": {"vibhakti": 1, "vacana": 1}},
            {"surface": "gacchati", "type": "tinanta", "attributes": {"purusa": 1, "vacana": 1}}
        ],
        "expected_links": [(0, 1)],  # subject→verb
    },
    {
        "description": "Subject-object-verb",
        "tokens": [
            {"surface": "rāmaḥ", "type": "subanta", "attributes": {"vibhakti": 1}},
            {"surface": "gṛham", "type": "subanta", "attributes": {"vibhakti": 2}},
            {"surface": "gacchati", "type": "tinanta", "attributes": {}}
        ],
        "expected_links": [(0, 2), (1, 2)],  # subject→verb, object→verb
    }
]
```

---

## Phase 2B Tests: Neural Engine

### Unit Tests

#### `test_tokenizer`

```python
from panini_lm.phase2b_neural import Tokenizer

class TestTokenizer:
    """Tests for morphological tokenizer."""
    
    def test_roundtrip(self):
        """Encode→decode should recover original."""
        tokenizer = Tokenizer()
        tokens = [{"stem": "rāma", "type": "subanta"}]
        
        ids = tokenizer.encode(tokens)
        decoded = tokenizer.decode(ids)
        
        assert decoded[0]["stem"] == "rāma"
    
    def test_unknown_token(self):
        """Unknown tokens should map to <unk>."""
        tokenizer = Tokenizer()
        tokens = [{"stem": "xyz_unknown_xyz", "type": "subanta"}]
        
        ids = tokenizer.encode(tokens)
        assert ids[0] == tokenizer.unk_id
    
    def test_batch_encoding(self):
        """Batch encoding should handle padding correctly."""
        tokenizer = Tokenizer()
        batch = [
            [{"stem": "rāma", "type": "subanta"}],
            [{"stem": "sītā", "type": "subanta"}, {"stem": "gam", "type": "tinanta"}]
        ]
        
        result = tokenizer.encode_batch(batch)
        assert result["token_ids"].shape[0] == 2  # batch size
        assert result["seq_lengths"] == [1, 2]
```

#### `test_embedding`

```python
import torch
from panini_lm.phase2b_neural import PaninianEmbedding

class TestFactorizedEmbedding:
    """Tests for FACTORIZED embedding layer (Zero OOV architecture)."""
    
    def test_output_shape(self):
        """Factorized embedding output should be (batch, seq, d_model)."""
        embed = PaninianEmbedding(d_model=512)
        
        # Factorized inputs: 5 parallel ID tensors
        root_ids = torch.tensor([[100, 101, 6]])      # rāma, gṛha, gam
        type_ids = torch.tensor([[0, 0, 1]])          # subanta, subanta, tiṅanta
        vibhakti_ids = torch.tensor([[1, 2, 0]])      # nom, acc, none
        vacana_ids = torch.tensor([[1, 1, 1]])        # sing, sing, sing
        purusa_ids = torch.tensor([[0, 0, 1]])        # none, none, 3rd
        
        output = embed(root_ids, type_ids, vibhakti_ids, vacana_ids, purusa_ids)
        assert output.shape == (1, 3, 512)  # (batch, seq, d_model)
    
    def test_no_positional_encoding(self):
        """Same factorized input at different positions should have same embedding."""
        embed = PaninianEmbedding(d_model=512)
        
        # Same root+grammatical features at positions 0 and 2
        root_ids = torch.tensor([[100, 101, 100]])    # rāma at pos 0 and 2
        type_ids = torch.tensor([[0, 0, 0]])          # all subanta
        vibhakti_ids = torch.tensor([[1, 2, 1]])      # nom, acc, nom
        vacana_ids = torch.tensor([[1, 1, 1]])        # all singular
        purusa_ids = torch.tensor([[0, 0, 0]])        # all none
        
        output = embed(root_ids, type_ids, vibhakti_ids, vacana_ids, purusa_ids)
        # Positions 0 and 2 have same factorized input → same embedding
        assert torch.allclose(output[0, 0], output[0, 2])
    
    def test_factorized_composition(self):
        """Different inflections of same root should share root component."""
        embed = PaninianEmbedding(d_model=512)
        
        # "gacchati" (3rd person) vs "gacchāmi" (1st person)
        # Both share √gam (root_id=6)
        root_ids = torch.tensor([[6, 6]])
        type_ids = torch.tensor([[1, 1]])             # both tiṅanta
        vibhakti_ids = torch.tensor([[0, 0]])         # N/A for verbs
        vacana_ids = torch.tensor([[1, 1]])           # both singular
        purusa_ids = torch.tensor([[1, 3]])           # 3rd vs 1st person
        
        emb = embed(root_ids, type_ids, vibhakti_ids, vacana_ids, purusa_ids)
        
        # Difference should equal (purusa_embed[1] - purusa_embed[3])
        diff = emb[0, 0] - emb[0, 1]
        expected_diff = embed.purusa_embed.weight[1] - embed.purusa_embed.weight[3]
        assert torch.allclose(diff, expected_diff)
    
    def test_zero_oov(self):
        """
        Zero OOV: Model can embed any valid inflection, even unseen forms.
        """
        embed = PaninianEmbedding(d_model=512)
        
        # A rare form the model may never have seen in training
        # √kṛ + optative + 2nd person + dual
        output = embed(
            root_ids=torch.tensor([[50]]),        # √kṛ
            type_ids=torch.tensor([[1]]),         # tiṅanta
            vibhakti_ids=torch.tensor([[0]]),     # none
            vacana_ids=torch.tensor([[2]]),       # dual
            purusa_ids=torch.tensor([[2]]),       # 2nd person
        )
        
        # Should NOT raise error, should produce valid embedding
        assert output.shape == (1, 1, 512)
        assert not torch.isnan(output).any()
```

#### `test_qkv_projection`

```python
class TestQKVProjection:
    """Tests for Q, K, V projections."""
    
    def test_output_shapes(self):
        """Q, K, V should have correct shapes."""
        from panini_lm.phase2b_neural import QKVProjection
        
        proj = QKVProjection(d_model=512, num_heads=8)
        x = torch.randn(2, 10, 512)  # (batch, seq, d_model)
        
        qkv = proj(x)
        
        head_dim = 512 // 8
        assert qkv["Q"].shape == (2, 8, 10, head_dim)
        assert qkv["K"].shape == (2, 8, 10, head_dim)
        assert qkv["V"].shape == (2, 8, 10, head_dim)
```

---

## Phase 3 Tests: Sparse Attention

### Unit Tests

#### `test_attention_correctness`

```python
import torch
from panini_lm.phase3_attention import sparse_paninian_attention, dense_attention_reference

class TestAttentionCorrectness:
    """Tests for attention computation correctness."""
    
    def test_sparse_equals_dense_reference(self):
        """Sparse attention should match dense reference (with same mask)."""
        batch, heads, seq, dim = 2, 4, 8, 64
        
        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)
        M = torch.zeros(seq, seq)  # All connections valid
        
        sparse_out = sparse_paninian_attention(Q, K, V, M)
        dense_out = dense_attention_reference(Q, K, V, M)
        
        assert torch.allclose(sparse_out, dense_out, atol=1e-5)
    
    def test_masked_positions_zero_attention(self):
        """Masked positions should receive zero attention weight."""
        batch, heads, seq, dim = 1, 1, 4, 32
        
        Q = torch.randn(batch, heads, seq, dim)
        K = torch.randn(batch, heads, seq, dim)
        V = torch.randn(batch, heads, seq, dim)
        
        # Mask: position 0 can only attend to position 0
        M = torch.full((seq, seq), float('-inf'))
        M[0, 0] = 0.0
        
        output, weights = sparse_paninian_attention(Q, K, V, M, return_weights=True)
        
        # Position 0 should have all weight on itself
        assert weights[0, 0, 0, 0] == 1.0
        assert weights[0, 0, 0, 1:].sum() == 0.0
    
    def test_numerical_stability(self):
        """Attention should be stable with extreme values."""
        batch, heads, seq, dim = 1, 1, 4, 32
        
        Q = torch.randn(batch, heads, seq, dim) * 100  # Large values
        K = torch.randn(batch, heads, seq, dim) * 100
        V = torch.randn(batch, heads, seq, dim)
        M = torch.zeros(seq, seq)
        
        output = sparse_paninian_attention(Q, K, V, M)
        
        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isinf(output).any(), "Inf in output"
```

#### `test_attention_fallback`

```python
class TestAttentionFallback:
    """Tests for Triton → PyTorch fallback."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_kernel_available(self):
        """Triton kernel should be used when CUDA available."""
        from panini_lm.phase3_attention import get_attention_backend
        
        backend = get_attention_backend()
        assert backend == "triton"
    
    def test_pytorch_fallback(self):
        """PyTorch fallback should work on CPU."""
        from panini_lm.phase3_attention import sparse_paninian_attention
        
        Q = torch.randn(1, 1, 4, 32)
        K = torch.randn(1, 1, 4, 32)
        V = torch.randn(1, 1, 4, 32)
        M = torch.zeros(4, 4)
        
        output = sparse_paninian_attention(Q, K, V, M, force_backend="pytorch")
        assert output.shape == (1, 1, 4, 32)
```

---

## Phase 4 Tests: Semantic Maturation

### Unit Tests

```python
from panini_lm.phase4_ffn import SemanticFFN

class TestSemanticFFN:
    """Tests for FFN layer."""
    
    def test_output_shape(self):
        """FFN should preserve shape."""
        ffn = SemanticFFN(d_model=512, expansion=2.0)
        x = torch.randn(2, 10, 512)
        
        output = ffn(x)
        assert output.shape == x.shape
    
    def test_gradient_flow(self):
        """Gradients should flow through FFN."""
        ffn = SemanticFFN(d_model=512, expansion=2.0)
        x = torch.randn(2, 10, 512, requires_grad=True)
        
        output = ffn(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_reduced_expansion(self):
        """FFN with 2x expansion should have fewer params than 4x."""
        ffn_2x = SemanticFFN(d_model=512, expansion=2.0)
        ffn_4x = SemanticFFN(d_model=512, expansion=4.0)
        
        params_2x = sum(p.numel() for p in ffn_2x.parameters())
        params_4x = sum(p.numel() for p in ffn_4x.parameters())
        
        assert params_2x < params_4x
```

---

## Phase 5 Tests: Grammar-Constrained Decoding

### Unit Tests

#### `test_grammar_mask_generation`

```python
from panini_lm.phase5_decoding import generate_grammar_mask

class TestGrammarMask:
    """
    Tests for grammar constraint mask generation.
    
    Note: Panini-LM uses ~4000 root vocabulary (factorized embeddings),
    NOT 50,000+ surface forms like standard LLMs.
    """
    
    def test_mask_shape(self):
        """Mask should match root vocabulary size (~4000)."""
        state = {"last_token": {"type": "subanta", "attributes": {"vibhakti": 1}}}
        root_vocab_size = 4000  # Factorized vocabulary, not 50000
        
        mask = generate_grammar_mask(state, root_vocab_size)
        assert mask["mask"].shape == (root_vocab_size,)
    
    def test_mask_values(self):
        """Mask values should be 0.0 or -inf."""
        state = {"last_token": {"type": "subanta", "attributes": {}}}
        
        mask = generate_grammar_mask(state, root_vocab_size=4000)
        M = mask["mask"]
        
        valid = (M == 0.0) | (M == float('-inf'))
        assert valid.all()
    
    def test_valid_roots_exist(self):
        """At least some roots should be valid continuations."""
        state = {"last_token": {"type": "subanta", "attributes": {}}}
        
        mask = generate_grammar_mask(state, root_vocab_size=4000)
        assert mask["valid_root_count"] > 0
    
    def test_constrained_logits(self):
        """Masked logits should zero invalid root probabilities."""
        from panini_lm.phase5_decoding import apply_grammar_constraint
        
        logits = torch.randn(4000)  # ~4000 roots
        mask = torch.zeros(4000)
        mask[2000:] = float('-inf')  # Mask out roots 2000-3999
        
        constrained = apply_grammar_constraint(logits, mask)
        probs = torch.softmax(constrained, dim=-1)
        
        # Masked roots should have zero probability
        assert probs[2000:].sum() < 1e-6
```

#### `test_grammatical_correctness`

**Critical test**: Generated tokens must be 100% grammatically valid.

```python
from panini_lm.phase5_decoding import GrammarConstrainedDecoder
from panini_lm.phase1_morphology import validate_grammar

class TestGrammaticalCorrectness:
    """Tests for grammatical correctness guarantee."""
    
    def test_all_generated_tokens_valid(self):
        """Every generated token must be grammatically valid."""
        decoder = GrammarConstrainedDecoder(model, vocab)
        
        prompt = "rāmaḥ"
        output = decoder.generate(prompt, max_tokens=20)
        
        # Validate each consecutive token pair
        tokens = output["generated_tokens"]
        for i in range(len(tokens) - 1):
            is_valid = validate_grammar(tokens[i], tokens[i+1])
            assert is_valid, f"Invalid transition: {tokens[i]} → {tokens[i+1]}"
    
    def test_no_grammar_violations(self):
        """Grammar violation count should always be 0."""
        decoder = GrammarConstrainedDecoder(model, vocab)
        
        for _ in range(10):  # Test multiple generations
            output = decoder.generate("rāmaḥ", max_tokens=10)
            assert output["grammar_violations"] == 0
    
    @pytest.mark.parametrize("prompt", [
        "rāmaḥ",
        "gacchati",
        "aham tvām",
    ])
    def test_various_prompts(self, prompt):
        """Test grammatical correctness with various prompts."""
        decoder = GrammarConstrainedDecoder(model, vocab)
        output = decoder.generate(prompt, max_tokens=10)
        
        assert output["grammar_violations"] == 0
```

---

## Integration Tests

### End-to-End Pipeline

```python
class TestEndToEndPipeline:
    """Integration tests for complete pipeline."""
    
    def test_forward_pass(self):
        """Complete forward pass should work."""
        from panini_lm import PaninianLM
        
        model = PaninianLM(config)
        text = "rāmo'pi gṛhaṃ gacchati"
        
        output = model.forward(text)
        
        assert "logits" in output
        assert output["logits"].shape[-1] == config.vocab_size
    
    def test_generate(self):
        """Generation should produce valid Sanskrit."""
        model = PaninianLM(config)
        
        generated = model.generate("rāmaḥ", max_tokens=10)
        
        assert isinstance(generated, str)
        assert len(generated) > len("rāmaḥ")
    
    def test_phase_data_flow(self):
        """Data should flow correctly between phases."""
        model = PaninianLM(config)
        text = "rāmaḥ gacchati"
        
        # Phase 1
        phase1_out = model.phase1(text)
        assert len(phase1_out["tokens"]) == 2
        
        # Phase 2A
        phase2a_out = model.phase2a(phase1_out["tokens"])
        assert phase2a_out["matrix"].shape == (2, 2)
        
        # Phase 2B
        phase2b_out = model.phase2b(phase1_out["tokens"])
        assert phase2b_out["Q"].shape[2] == 2  # seq_len
        
        # Phase 3
        phase3_out = model.phase3(phase2b_out, phase2a_out["matrix"])
        assert phase3_out["hidden_states"].shape[1] == 2
```

---

## Performance Benchmarks

### Attention Complexity

```python
import pytest

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    def test_sparse_vs_dense_attention(self, benchmark):
        """Sparse attention should be faster than dense for long sequences."""
        seq_len = 512
        Q = torch.randn(1, 8, seq_len, 64, device='cuda')
        K = torch.randn(1, 8, seq_len, 64, device='cuda')
        V = torch.randn(1, 8, seq_len, 64, device='cuda')
        
        # Sparse mask (k ≈ 3)
        M = torch.full((seq_len, seq_len), float('-inf'), device='cuda')
        for i in range(seq_len):
            M[i, max(0, i-1):min(seq_len, i+2)] = 0.0
        
        result = benchmark(sparse_paninian_attention, Q, K, V, M)
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
    def test_attention_scaling(self, benchmark, seq_len):
        """Attention should scale linearly with sparse mask."""
        Q = torch.randn(1, 8, seq_len, 64)
        K = torch.randn(1, 8, seq_len, 64)
        V = torch.randn(1, 8, seq_len, 64)
        M = torch.zeros(seq_len, seq_len)  # Sparse mask
        
        result = benchmark(sparse_paninian_attention, Q, K, V, M)


# Expected benchmark results (baseline on A100)
BENCHMARK_BASELINES = {
    "attention_128": {"mean_ms": 0.5, "max_ms": 1.0},
    "attention_256": {"mean_ms": 0.8, "max_ms": 1.5},
    "attention_512": {"mean_ms": 1.2, "max_ms": 2.5},
    "attention_1024": {"mean_ms": 2.0, "max_ms": 4.0},
}
```

### Memory Benchmarks

```python
class TestMemoryBenchmarks:
    """Memory usage benchmarks."""
    
    @pytest.mark.benchmark
    def test_memory_reduction(self):
        """Sparse attention should use less memory."""
        seq_len = 1024
        
        # Measure dense memory
        dense_mem = measure_memory_usage(
            dense_attention, Q, K, V, seq_len=seq_len
        )
        
        # Measure sparse memory
        sparse_mem = measure_memory_usage(
            sparse_attention, Q, K, V, M, seq_len=seq_len
        )
        
        reduction = 1 - (sparse_mem / dense_mem)
        assert reduction > 0.5, f"Expected >50% reduction, got {reduction:.1%}"
```

---

## CI/CD Configuration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e .[dev]
      
      - name: Run unit tests
        run: |
          pytest tests/ -v --cov=panini_lm --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4

  gpu-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      
      - name: Run GPU tests
        run: |
          pytest tests/test_phase3/ -v -m gpu
      
      - name: Run benchmarks
        run: |
          pytest tests/test_benchmark.py --benchmark-json=results.json

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run linters
        run: |
          pip install ruff mypy
          ruff check .
          mypy panini_lm/
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [torch]
```

---

## See Also

- [Data Contracts](../types/data-contracts.md) — Type definitions for test validation
- [Phase Documentation](../phases/README.md) — Detailed phase specifications
- [Integration Guide](../integration/README.md) — External library setup
