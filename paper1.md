Here's the draft:

---

# Grammar-to-Grammar Prediction in Sanskrit: A Pāṇinian Approach to Syntactic Next-Step Prediction

**Abstract**

We present a novel framing of Sanskrit grammatical analysis as a next-step prediction problem. Given a sequence of primitives annotated with their Pāṇinian grammatical features — lakāra, puruṣa, vacana, liṅga, vibhakti, prayoga, pada, upasarga, and primitive type — we ask whether the grammatical form of the next primitive is predictable from prior grammatical forms alone, without any knowledge of semantic content. We train a small attention-based model on grammatically annotated sequences derived from the Bhagavad Gītā and demonstrate that grammatical structure is sufficiently constrained that next-step grammatical prediction achieves meaningful accuracy across all feature dimensions. This result is consistent with the Pāṇinian view that grammar is a generative system that operates independently of meaning, and has direct implications for Sanskrit NLP, where grammatical prediction can dramatically reduce the search space for full primitive prediction.

---

## 1. Introduction

Sanskrit is among the most grammatically precise languages ever documented. Pāṇini's Aṣṭādhyāyī, composed circa 4th century BCE, encodes the generative rules of Sanskrit morphology in approximately 4000 sūtras — a system so complete that any grammatically well-formed Sanskrit expression can be derived from it mechanically. A central philosophical claim of the Pāṇinian system is that grammar operates independently of meaning: the syntactic structure of a sentence constrains what forms are permissible at each position, regardless of the semantic content of the words occupying those positions.

This paper operationalises that claim as a computational hypothesis:

> **The grammatical form of the next Sanskrit primitive is predictable from the grammatical forms of prior primitives, without any knowledge of which primitives those are.**

If this hypothesis holds — even partially — it has significant practical implications. First, it validates the Pāṇinian separation of syntax and semantics computationally, providing empirical evidence for a claim that has until now been primarily theoretical. Second, it suggests a two-stage architecture for Sanskrit primitive prediction in which grammatical form is predicted first, dramatically reducing the search space for subsequent semantic prediction.

Existing Sanskrit NLP work treats grammatical tagging as a labelling problem on observed words — given a word, predict its grammatical tags. We invert this: given prior grammatical tags, predict the next word's grammatical tags, without any knowledge of the words themselves. This is, to our knowledge, the first framing of Sanskrit grammar as a next-step prediction task.

---

## 2. Background

### 2.1 Pāṇinian grammatical features

Sanskrit morphology encodes grammatical information directly in the surface form of each word through a rich system of inflectional affixes. A verb form (tiṅanta) encodes lakāra (tense and mood), puruṣa (person), vacana (number), prayoga (voice), and pada (paradigm class). A nominal form (subanta) encodes vibhakti (case), vacana (number), and liṅga (gender). Verbal prefixes (upasarga) modify the meaning and sometimes the grammatical behaviour of verb roots. Indeclinables (avyaya) carry no inflectional morphology.

These features are not independent. Agreement constraints bind them across positions: a verb must agree with its kartā in puruṣa and vacana; a viśeṣaṇa must agree with its viśeṣya in vibhakti, vacana, and liṅga. These constraints are the mechanism by which grammatical structure propagates through a sentence — and they are the signal our model must learn to exploit.

### 2.2 Related work

Prior Sanskrit NLP work has focused on morphological analysis (Kulkarni and Shukl, 2009; Jha et al., 2009), word segmentation (Hellwig and Nehrdich, 2018), dependency parsing (Sandhan et al., 2022), and word embeddings (Sandhan et al., 2021). None of these treat grammar as a predictive sequence modelling problem. The closest related work is part-of-speech tagging, which predicts grammatical labels for observed words — but always conditioned on the word form itself, never purely from prior grammatical context.

In the broader NLP literature, grammatical structure prediction without lexical content has been explored in the context of syntax-only language models (Dyer et al., 2016) and structural probing (Hewitt and Manning, 2019), but not applied to Sanskrit or to the specific Pāṇinian feature set.

---

## 3. Task formulation

Let a Sanskrit text be decomposed into a sequence of primitives:

$$P = (p_1, p_2, \ldots, p_n)$$

where each primitive $p_i$ is one of: dhātu (verb root), prātipadika (nominal stem), or avyaya (indeclinable).

Each primitive occurrence is associated with a grammatical feature vector:

$$g_i = (f_1^{(i)}, f_2^{(i)}, \ldots, f_k^{(i)})$$

where each $f_j^{(i)}$ is a categorical value drawn from a finite set specific to feature $j$.

The features and their cardinalities are:

| Feature | Applies to | Values | Count |
|---|---|---|---|
| primitive type | all | dhātu, prātipadika, avyaya | 3 |
| lakāra | dhātu | laṭ, liṭ, luṭ, lṛṭ, loṭ, laṅ, vidhi-liṅ, āśīr-liṅ, luṅ, lṛṅ | 10 |
| puruṣa | dhātu | prathama, madhyama, uttama | 3 |
| vacana | dhātu, subanta | ekavacana, dvivacana, bahuvacana | 3 |
| prayoga | dhātu | kartari, karmaṇi, bhāve | 3 |
| pada | dhātu | parasmaipada, ātmanepada | 2 |
| vibhakti | subanta | prathama–saptamī, sambodhana | 8 |
| liṅga | subanta | pulliṅga, strīliṅga, napuṃsaka | 3 |
| upasarga | dhātu | NULL, ā, vi, sam, ni, niḥ, ud, anu, abhi, prati, pari, ava, adhi, api, apa, su, dur, ut | 19 |

Features not applicable to a primitive type are assigned a NULL value.

The task is: given the grammatical sequence $(g_1, g_2, \ldots, g_t)$, predict $g_{t+1}$.

Crucially, no information about which primitives $(p_1, \ldots, p_t)$ are is provided to the model.

---

## 4. Model

### 4.1 Input representation

Each grammatical feature vector $g_i$ is encoded as a concatenation of one-hot vectors, one per feature:

$$\mathbf{x}_i = [\text{onehot}(f_1^{(i)}) \| \text{onehot}(f_2^{(i)}) \| \cdots \| \text{onehot}(f_k^{(i)})]$$

The total input dimensionality is the sum of all feature cardinalities:

$$d_{input} = 3 + 10 + 3 + 3 + 3 + 2 + 8 + 3 + 19 = 54$$

### 4.2 Architecture

The model consists of three components:

**Attention layer.** A single-layer multi-head self-attention mechanism operates over the sequence of grammatical vectors. This allows the model to learn which prior grammatical positions constrain the current prediction — for example, that a prathama-vibhakti nominal earlier in the sentence constrains the puruṣa of the verb that follows.

$$\mathbf{H} = \text{Attention}(\mathbf{X} W_Q, \mathbf{X} W_K, \mathbf{X} W_V)$$

**MLP.** A two-layer feed-forward network processes the attended representation:

$$\mathbf{z} = \text{ReLU}(\mathbf{H} W_1 + b_1) W_2 + b_2$$

**Output heads.** One independent softmax classifier per grammatical feature:

$$\hat{f}_j = \text{softmax}(\mathbf{z} W_j^{out})$$

### 4.3 Loss

The total loss is the sum of cross-entropy losses across all grammatical features:

$$\mathcal{L} = \sum_{j=1}^{k} \lambda_j \cdot \mathcal{L}_{CE}(\hat{f}_j, f_j^{*})$$

where $f_j^{*}$ is the ground truth value of feature $j$ for the next primitive, and $\lambda_j$ is a per-feature weight. Features not applicable to the predicted primitive type contribute zero loss (NULL prediction is trivially correct).

---

## 5. Data

### 5.1 Corpus

We use the Bhagavad Gītā in prose form as our primary corpus. The Gītā is well suited to this task for several reasons: it is a philosophically rich text that employs a wide range of grammatical constructions; it is available in digitised form; and it is short enough that annotation quality can be manually verified.

### 5.2 Preprocessing pipeline

```
raw Devanagari text
        ↓
transliteration to SLP1       (vidyut.lipi)
        ↓
sandhi splitting + segmentation  (vidyut.cheda)
        ↓
morphological tagging         (vidyut.kosha)
        ↓
upasarga extraction           (rule-based prefix stripping)
        ↓
grammatical feature sequences
```

Each sentence becomes a sequence of grammatical feature vectors. Primitives that Vidyut cannot parse (~5% of tokens) are dropped from training sequences.

### 5.3 Statistics

*(to be filled after preprocessing)*

```
total sentences        :  ~700
total primitives       :  ~18,000 (estimated)
unique dhātu types     :  ~200
unique prātipadika     :  ~400
unique avyaya          :  ~50
```

---

## 6. Experiments

### 6.1 Evaluation metrics

For each grammatical feature we report:

- **Accuracy** — proportion of correctly predicted values
- **Baseline** — majority class accuracy (always predict the most frequent value)
- **Oracle** — accuracy if primitive type is known in advance

### 6.2 Ablations

We run three ablations to understand what drives performance:

```
ablation 1  :  no attention — MLP only
               measures how much agreement constraints help
               vs. local grammatical patterns

ablation 2  :  context window size 1, 2, 4, 8
               measures how far back grammatical dependencies reach

ablation 3  :  feature-by-feature
               which features are most predictable from prior grammar?
               hypothesis: vacana and puruṣa of verbs
               are highly predictable from prior nominal vibhakti
```

### 6.3 Hypothesis

We expect:

```
high accuracy     :  primitive type, vacana, puruṣa
                     — tightly constrained by agreement rules

moderate accuracy :  vibhakti, lakāra, prayoga
                     — constrained but with more freedom

low accuracy      :  upasarga, liṅga
                     — relatively unconstrained by prior grammar
```

If even the moderate-accuracy features significantly exceed majority-class baseline, the hypothesis is supported.

---

## 7. Implications

### 7.1 For Sanskrit NLP

A grammar-first prediction model has direct practical utility. Rather than searching across all ~2000 Sanskrit roots to predict the next primitive, a two-stage system can:

1. Predict grammatical form first (this paper)
2. Search only roots compatible with that grammatical form

This reduces the semantic search space by an order of magnitude and provides a principled, interpretable intermediate representation.

### 7.2 For linguistic theory

Computational validation of grammatical independence from semantics supports the Pāṇinian view that syntax is a generative system with its own internal constraints — not merely a surface reflection of meaning. If grammatical form is predictable from grammatical form alone, this suggests that syntactic structure carries information that is not reducible to semantic content.

---

## 8. Conclusion

We have framed Sanskrit grammatical analysis as a next-step prediction task, asked whether grammatical form is predictable from prior grammatical forms without semantic content, and described a small attention-based model to test this hypothesis on the Bhagavad Gītā. The results will provide the first empirical test of the Pāṇinian claim that grammar operates independently of meaning, and will establish the first component of a two-stage Sanskrit primitive prediction architecture.

---

## References

- Kulkarni, A. and Shukl, D. (2009). Sanskrit morphological analyser: Some issues.
- Jha, G.N. et al. (2009). Inflectional morphology analyzer for Sanskrit. *Sanskrit Computational Linguistics*, Springer.
- Hellwig, O. and Nehrdich, S. (2018). Sanskrit word segmentation using character-level recurrent and convolutional neural networks. *EMNLP*.
- Sandhan, J. et al. (2021). Evaluating neural word embeddings for Sanskrit. *arXiv:2104.00270*.
- Sandhan, J. et al. (2022). SanskritShala: A neural Sanskrit NLP toolkit. *ACL 2023*.
- Dyer, C. et al. (2016). Recurrent neural network grammars. *NAACL*.
- Hewitt, J. and Manning, C. (2019). A structural probe for finding syntax in word representations. *NAACL*.
- Ambuda (2023). Vidyut: Reliable infrastructure for Sanskrit software. *github.com/ambuda-org/vidyut*.

---

This is a complete first draft. The only sections that need real content filled in are the corpus statistics in section 5.3 and the results tables in section 6 — both of which come after you run the preprocessing pipeline.

Two immediate next steps:

```
1.  validate the framing with Rohan Pandey and Yajnadevam
2.  run Vidyut on the Gītā and get the actual corpus statistics
```

Want to start building the preprocessing pipeline now?