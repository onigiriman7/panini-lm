# Panini-LM Glossary

> Definitions of Sanskrit grammar terminology and architecture concepts.  
> Terms are alphabetized within each category.

---

## Table of Contents

- [Sanskrit Grammar Terms](#sanskrit-grammar-terms)
- [Token Types](#token-types)
- [Kāraka (Semantic Roles)](#kāraka-semantic-roles)
- [Vibhakti (Case Endings)](#vibhakti-case-endings)
- [Architecture Terms](#architecture-terms)
- [Library & Tool Terms](#library--tool-terms)

---

## Sanskrit Grammar Terms

### Aṣṭādhyāyī
**अष्टाध्यायी** — "Eight Chapters"

Pāṇini's foundational Sanskrit grammar treatise (~4th century BCE). Contains approximately 4,000 sūtras (rules) organized into eight chapters. Forms the mathematical basis for the symbolic engine in Panini-LM.

**Related**: [Phase 2A — Symbolic Engine](phases/phase2a-symbolic.md)

---

### Dhātu
**धातु** — "Root"

The verbal root from which verb forms are derived. Example: √गम् (gam) "to go" → गच्छति (gacchati) "goes".

**In Panini-LM**: Stored in `MorphToken.stem` for tiṅanta tokens.

---

### Lakāra
**लकार** — "Tense/Mood marker"

The tense-mood-aspect category of a verb. Sanskrit has 10 lakāras:

| Lakāra | Name | Meaning | Example |
|--------|------|---------|---------|
| लट् (laṭ) | Present | Present tense | गच्छति (gacchati) |
| लिट् (liṭ) | Perfect | Past completed | जगाम (jagāma) |
| लुट् (luṭ) | Periphrastic future | Will do | गन्ता (gantā) |
| लृट् (lṛṭ) | Simple future | Will go | गमिष्यति (gamiṣyati) |
| लेट् (leṭ) | Vedic subjunctive | May go | — |
| लोट् (loṭ) | Imperative | Go! | गच्छतु (gacchatu) |
| लङ् (laṅ) | Imperfect | Was going | अगच्छत् (agacchat) |
| लिङ् (liṅ) | Optative/Potential | Should go | गच्छेत् (gacchet) |
| लुङ् (luṅ) | Aorist | Went (remote) | अगमत् (agamat) |
| लृङ् (lṛṅ) | Conditional | Would go | अगमिष्यत् (agamiṣyat) |

**In Panini-LM**: Stored in `MorphAttributes.lakara`.

---

### Liṅga
**लिङ्ग** — "Gender"

Grammatical gender of nominals:

| Code | Sanskrit | English |
|------|----------|---------|
| `m` | पुंलिङ्ग (puṃliṅga) | Masculine |
| `f` | स्त्रीलिङ्ग (strīliṅga) | Feminine |
| `n` | नपुंसकलिङ्ग (napuṃsakaliṅga) | Neuter |

**In Panini-LM**: Stored in `MorphAttributes.linga`.

---

### Pada
**पद** — "Word"

A grammatically complete word unit after sandhi resolution. The basic unit of morphological analysis.

**In Panini-LM**: Phase 1 outputs a list of padas as `MorphToken` objects.

---

### Prakṛti
**प्रकृति** — "Base/Stem"

The base form of a word before affixes are added. For nominals, this is the prātipadika (nominal stem); for verbs, the dhātu (root).

**In Panini-LM**: Stored in `MorphToken.stem`.

---

### Pratyaya
**प्रत्यय** — "Affix/Suffix"

Grammatical suffixes added to stems to create inflected forms. Includes:
- **Sup** (सुप्): Case endings for nominals
- **Tiṅ** (तिङ्): Personal endings for verbs
- **Kṛt** (कृत्): Primary derivational suffixes
- **Taddhita** (तद्धित): Secondary derivational suffixes

**In Panini-LM**: Analyzed during Phase 1, influences `MorphAttributes`.

---

### Puruṣa
**पुरुष** — "Person"

Grammatical person for verbs:

| Code | Sanskrit | English | Example |
|------|----------|---------|---------|
| 1 | प्रथम (prathama) | 3rd person | गच्छति (gacchati) "he/she goes" |
| 2 | मध्यम (madhyama) | 2nd person | गच्छसि (gacchasi) "you go" |
| 3 | उत्तम (uttama) | 1st person | गच्छामि (gacchāmi) "I go" |

> **Note**: Sanskrit counts person inversely from English convention.

**In Panini-LM**: Stored in `MorphAttributes.purusa`. Critical for subject-verb agreement in Phase 2A.

---

### Samāsa
**समास** — "Compound"

Compound word formation where multiple stems combine into a single word. Types include:
- **Tatpuruṣa**: Determinative (राजपुत्र = king's son)
- **Dvandva**: Copulative (रामलक्ष्मणौ = Rāma and Lakṣmaṇa)
- **Bahuvrīhi**: Possessive (महाबाहु = having great arms)
- **Avyayībhāva**: Adverbial (उपकूलम् = near the bank)

**In Panini-LM**: Resolved during Phase 1 sandhi/samāsa resolution.

---

### Sandhi
**सन्धि** — "Junction/Joining"

Euphonic sound changes at word boundaries or morpheme junctions. Types:
- **Svara-sandhi**: Vowel sandhi (अ + इ → ए)
- **Vyañjana-sandhi**: Consonant sandhi (तत् + च → तच्च)
- **Visarga-sandhi**: Visarga changes (रामः + अपि → रामोऽपि)

**In Panini-LM**: Resolved during Phase 1 before morphological analysis.

**Example**:
```
Input:  "rāmo'pi gṛhaṃ gacchati"
Output: ["rāmaḥ", "api", "gṛham", "gacchati"]
```

---

### Sūtra
**सूत्र** — "Aphorism/Rule"

A concise grammatical rule in the Aṣṭādhyāyī. Sūtras are designed for maximum brevity.

**Example**: "इको यणचि" (iko yaṇaci) — "i/u/ṛ/ḷ become y/v/r/l before a dissimilar vowel"

**In Panini-LM**: Sūtras are implemented as deterministic rules in Phase 2A.

---

### Vacana
**वचन** — "Number"

Grammatical number:

| Code | Sanskrit | English | Example |
|------|----------|---------|---------|
| 1 | एकवचन (ekavacana) | Singular | गच्छति (gacchati) "goes" |
| 2 | द्विवचन (dvivacana) | Dual | गच्छतः (gacchataḥ) "two go" |
| 3 | बहुवचन (bahuvacana) | Plural | गच्छन्ति (gacchanti) "they go" |

**In Panini-LM**: Stored in `MorphAttributes.vacana`. Critical for agreement in Phase 2A.

---

### Vibhakti
**विभक्ति** — "Case"

Grammatical case for nominals:

| Code | Sanskrit | English | Function | Example |
|------|----------|---------|----------|---------|
| 1 | प्रथमा (prathamā) | Nominative | Subject | रामः (rāmaḥ) |
| 2 | द्वितीया (dvitīyā) | Accusative | Object | रामम् (rāmam) |
| 3 | तृतीया (tṛtīyā) | Instrumental | By/with | रामेण (rāmeṇa) |
| 4 | चतुर्थी (caturthī) | Dative | To/for | रामाय (rāmāya) |
| 5 | पञ्चमी (pañcamī) | Ablative | From | रामात् (rāmāt) |
| 6 | षष्ठी (ṣaṣṭhī) | Genitive | Of | रामस्य (rāmasya) |
| 7 | सप्तमी (saptamī) | Locative | In/on | रामे (rāme) |
| 8 | सम्बोधन (sambodhana) | Vocative | Address | हे राम (he rāma) |

**In Panini-LM**: Stored in `MorphAttributes.vibhakti`. Maps to Kāraka roles in Phase 2A.

---

## Token Types

### Subanta
**सुबन्त** — "Ending in sup (case suffix)"

A nominal word form (noun, adjective, pronoun) with case endings.

**In Panini-LM**: `MorphToken.type = "subanta"`

**Attributes**: vibhakti, vacana, liṅga

---

### Tiṅanta
**तिङन्त** — "Ending in tiṅ (personal suffix)"

A finite verb form with person/number endings.

**In Panini-LM**: `MorphToken.type = "tinanta"`

**Attributes**: puruṣa, vacana, lakāra

---

### Avyaya
**अव्यय** — "Indeclinable"

Words that do not inflect: particles, conjunctions, interjections.

**Examples**: च (ca) "and", अपि (api) "also", न (na) "not"

**In Panini-LM**: `MorphToken.type = "avyaya"`

---

### Kṛdanta
**कृदन्त** — "Ending in kṛt (primary suffix)"

Verbal derivatives: participles, infinitives, gerunds.

**Examples**: गच्छन् (gacchan) "going", गन्तुम् (gantum) "to go"

**In Panini-LM**: `MorphToken.type = "krdanta"`

---

## Kāraka (Semantic Roles)

**कारक** — "Factor/Agent"

Semantic roles linking nominals to verbs. The Kāraka system maps syntactic cases to semantic functions.

| Kāraka | Sanskrit | Role | Typical Vibhakti |
|--------|----------|------|------------------|
| Kartā | कर्ता | Agent (doer) | 1 (Nominative) |
| Karma | कर्म | Patient (object) | 2 (Accusative) |
| Karaṇa | करण | Instrument | 3 (Instrumental) |
| Sampradāna | सम्प्रदान | Recipient | 4 (Dative) |
| Apādāna | अपादान | Source/Origin | 5 (Ablative) |
| Adhikaraṇa | अधिकरण | Location | 7 (Locative) |

**In Panini-LM**: Stored in `MorphAttributes.karaka`. Used in Phase 2A to determine grammatical validity of connections.

**Example**:
```
"rāmaḥ gṛhaṃ gacchati" (Rāma goes home)
- rāmaḥ: kartā (agent) — nominative
- gṛham: karma (goal) — accusative  
- gacchati: kriyā (action) — verb
```

---

## Architecture Terms

### Adjacency Matrix M
**隣接行列 M**

An (N × N) sparse tensor encoding grammatically valid attention pathways:
- `M[i,j] = 0.0` — Token i may attend to token j
- `M[i,j] = -∞` — Grammatically impossible connection

**Sparsity**: Typically k ≈ 2-3 valid connections per token.

**Related**: [Phase 2A — Symbolic Engine](phases/phase2a-symbolic.md), [Data Contracts](types/data-contracts.md)

---

### Block-Sparse Attention

GPU-optimized attention computation that skips loading and computing invalid blocks based on the adjacency matrix M.

**Implementation**: Triton kernel checks M before fetching K vectors.

**Related**: [Phase 3 — Sparse Attention](phases/phase3-attention.md), [Triton Integration](integration/triton.md)

---

### Dual-Track Architecture

Panini-LM's separation of:
1. **Symbolic Track** (Phase 2A): Deterministic grammatical structure → Matrix M
2. **Neural Track** (Phase 2B): Learned semantic embeddings → Q, K, V

The tracks merge in Phase 3 (Sparse Attention).

---

### Factorized Embeddings

**THE KEY INNOVATION** — Word embeddings constructed by summing morphological component vectors instead of direct lookup:

```python
E(word) = E(root) + E(type) + E(vibhakti) + E(vacana) + E(puruṣa)
```

**Benefits**:
- **Zero OOV**: Any valid inflection can be embedded compositionally
- **12× parameter reduction**: ~2M vs ~25M embedding parameters
- **Structural encoding**: Morphological knowledge preserved, not learned implicitly

**Example**:
```
E(gacchati) = E(√gam) + E(tiṅanta) + E(laṭ) + E(prathama) + E(eka-vacana)
E(gacchāmi) = E(√gam) + E(tiṅanta) + E(laṭ) + E(uttama)   + E(eka-vacana)
```

Both share the same root embedding E(√gam) — only the grammatical components differ.

**Related**: [Phase 2B — Neural Engine](phases/phase2b-neural.md), [Data Contracts](types/data-contracts.md)

---

### FactorizedTokenBatch

The input data structure to Phase 2B, containing 5 parallel ID tensors:

| Tensor | Description | Size |
|--------|-------------|------|
| `root_ids` | Root/stem IDs (semantic core) | ~4000 |
| `type_ids` | Token type (subanta/tiṅanta/etc.) | 7 |
| `vibhakti_ids` | Case (1-7, vocative, none) | 9 |
| `vacana_ids` | Number (sing/dual/plural/none) | 4 |
| `purusa_ids` | Person (3rd/2nd/1st/none) | 4 |

**Related**: [Data Contracts](types/data-contracts.md)

---

### Grammar-Constrained Decoding

Inference-time technique that masks grammatically impossible next tokens, guaranteeing 100% grammatical correctness.

**Formula**: `constrained_logits = raw_logits + grammar_mask`

Where `grammar_mask[i] = -∞` for impossible tokens.

**Related**: [Phase 5 — Decoding](phases/phase5-decoding.md)

---

### Position-Agnostic Embeddings

Embeddings without positional encoding (no RoPE, no sinusoidal). Enables native support for Sanskrit's free word order.

**Related**: [Phase 2B — Neural Engine](phases/phase2b-neural.md)

---

### Sparse Attention

Attention mechanism with O(N·k) complexity instead of O(N²), achieved by computing only grammatically valid attention scores.

**Formula**: `Attention(Q,K,V) = softmax(Sparse(QK^T/√d_k) + M) × V`

**Related**: [Phase 3 — Sparse Attention](phases/phase3-attention.md)

---

### Zero OOV

"Zero Out-Of-Vocabulary" — the property that Panini-LM can embed ANY valid Sanskrit inflection, even forms never seen during training.

Achieved through **Factorized Embeddings**: as long as the model knows the root (√gam) and the grammatical tags, it can construct the embedding for any inflected form.

**Contrast with standard LLMs**: Unknown words map to a single `[UNK]` token, losing all semantic/grammatical information.

**Related**: [Phase 2B — Neural Engine](phases/phase2b-neural.md), [Training Guide](training/README.md)

---

## Library & Tool Terms

### vidyut-prakriya

Open-source Sanskrit morphological analyzer written in Rust. Provides:
- Sandhi resolution
- Morphological analysis (stem extraction, attribute parsing)
- High performance via Rust compilation

**Python integration**: PyO3 bindings

**Repository**: [github.com/ambuda-org/vidyut](https://github.com/ambuda-org/vidyut)

**Related**: [vidyut Integration](integration/vidyut.md)

---

### sanskrit-heritage

Python-based Sanskrit analysis toolkit. Used as fallback when vidyut-prakriya is unavailable.

**Features**: Sandhi splitting, morphological lookup, dictionary access

**Related**: [sanskrit-heritage Integration](integration/sanskrit-heritage.md)

---

### samsadhani

University of Hyderabad's Sanskrit computational linguistics platform. Provides:
- Dependency parsing
- Kāraka relationship identification
- REST API access

**In Panini-LM**: Optional API for precomputing Kāraka links during training data preparation.

**Related**: [samsadhani Integration](integration/samsadhani.md)

---

### Triton

OpenAI's GPU kernel language for writing custom CUDA kernels in Python.

**In Panini-LM**: Used to implement block-sparse attention that skips invalid computations.

**Related**: [Triton Integration](integration/triton.md)

---

### PyO3

Rust-Python interoperability library for creating Python extensions from Rust code.

**In Panini-LM**: Used to wrap vidyut-prakriya for Python access.

---

## See Also

- [Documentation Index](INDEX.md)
- [Data Contracts](types/data-contracts.md)
- [Phase Overview](phases/README.md)
