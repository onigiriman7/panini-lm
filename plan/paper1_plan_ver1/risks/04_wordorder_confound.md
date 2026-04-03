# Risk 04: The Model May Learn Word-Order Statistics Instead of Grammar

## Severity: MEDIUM

## Which part of the paper this affects

### §5.1 — Corpus (The Gītā specifically)

> *"We use the Bhagavad Gītā in prose form as our primary corpus."*

The critical phrase is **"in prose form."** The Gītā has been rewritten from verse into prose — a process called **anvaya** in the Sanskrit tradition. Anvaya rearranges verse-order words into a natural grammatical reading order, which is close to standard SOV (Subject-Object-Verb).

This means our training data has **more regular word order** than typical Sanskrit text. The model has an easier time exploiting positional statistics because the positions have been artificially normalised.

### §6.3 — Hypothesis (What the model learns)

The paper expects:

> *"high accuracy: primitive type, vacana, puruṣa — tightly constrained by agreement rules"*

But the question is: **is this accuracy coming from agreement rule learning or from word-order pattern matching?**

---

## What the risk actually is

### Two different signals, same accuracy number

The model has access to two types of information:

**Signal A: Grammatical constraints (what we want to measure)**
```
Rule: If a prathamā-vibhakti subanta appears, and a tiṅanta follows, 
      the tiṅanta's vacana must match the subanta's vacana.

This is a genuine grammatical rule. It holds regardless of word order.
```

**Signal B: Positional/ordering statistics (a confound)**
```
Pattern: In our prose Gītā data, position 1 is usually a noun,
         positions 2-3 are usually more nouns or adverbs,
         and the verb usually comes at or near the end.

This is a statistical regularity of this specific corpus's word order.
```

Both signals produce above-baseline accuracy on the same features. The model's loss function doesn't distinguish between them. A model that achieves 80% accuracy on type prediction might be:

- 60% from grammatical constraints + 20% from word-order patterns
- 20% from grammatical constraints + 60% from word-order patterns
- 0% from grammatical constraints + 80% from word-order patterns

Without further analysis, we can't tell.

### Why the prose conversion makes this worse

Original Gītā verse (flexible word order):
```
dharmakṣetre kurukṣetre samavetā yuyutsavaḥ
māmakāḥ pāṇḍavāś caiva kim akurvata sañjaya
```

Prose conversion (regularised to near-SOV):
```
हे सञ्जय, धर्मक्षेत्रे कुरुक्षेत्रे समवेताः 
युयुत्सवः मामकाः पाण्डवाः च किम् अकुर्वन्।
```

The prose form imposes a consistent ordering pattern that the model can exploit without understanding grammar. In the original verse form, word positions are much more variable.

### What would happen with different word orders?

If Sanskrit truly has grammatically free word order, then the same grammatical content can appear in many orderings. The model's accuracy on agreement features should be tested against shuffled versions of the same sentences:

```
Original order:    [sub-prathamā-eka] [sub-dvitīyā-eka] [tiṅ-prathama-eka]
                   Type accuracy: 85%  Vacana accuracy: 78%

Shuffled order:    [tiṅ-prathama-eka] [sub-dvitīyā-eka] [sub-prathamā-eka]
                   Type accuracy: ??%  Vacana accuracy: ??%
```

If accuracy drops dramatically under shuffling → the model was relying on word-order statistics.
If accuracy on agreement features holds up → the model learned genuine grammatical constraints.

---

## How serious is this?

### This is a confound, not a flaw

The distinction matters:

- **Flaw**: "The paper's method doesn't work" — this is NOT the case
- **Confound**: "The paper's method works, but we can't tell why" — this IS the concern

A confound doesn't invalidate results — it limits the strength of conclusions. The paper claims to validate the Pāṇinian separation of syntax and semantics. If the signal is mostly word-order statistics, the conclusion becomes "word order in prose Sanskrit is regular" — true, but less interesting.

### Related: Anvaya as a theoretical concern

There's a deeper issue: the very existence of anvaya (prose reordering) in the Sanskrit tradition is actually **evidence FOR the hypothesis**. The fact that scholars can reorder any Sanskrit verse into prose order — purely by following grammatical cues — proves that grammar constrains word order. This means:

> Word-order regularity in Sanskrit prose is itself a consequence of grammatical structure.

So "learning word-order statistics" and "learning grammar" may not be as separable as they first appear. Still, the paper should address this explicitly.

---

## Specific paper sections to strengthen

### §5.1 — Acknowledge the anvaya/prose conversion

> *"Our corpus is the Gītā in prose form (anvaya). This normalisation produces more regular word order than original verse Sanskrit. While this simplifies the prediction task, it also means our results reflect a specific, regularised ordering convention. We note that this regularity is itself a grammatical phenomenon — anvaya is possible precisely because kāraka relations are recoverable from morphological markers — but results should be validated on verse-order and free-prose Sanskrit in future work."*

### §6.2 — Add the permutation test (most critical intervention)

This was introduced in Risk 01 but is worth detailing further as the single most important experiment for addressing this confound:

```
Experiment: Permutation test for word-order dependence

Method:
  1. Take training data as-is (prose order)
  2. Create a shuffled copy: within each sentence, randomly 
     permute the order of grammatical vectors
  3. Train identical models on both versions
  4. Report per-feature accuracy for both

Feature-level interpretation:
  Feature          Prose   Shuffled   Interpretation
  ─────────────    ─────   ────────   ──────────────
  type             85%     55%        Word-order dependent
  vacana           78%     73%        Mostly grammatical
  puruṣa           82%     78%        Mostly grammatical
  vibhakti         60%     45%        Partially word-order dependent
  lakāra           58%     56%        Weakly order-dependent
  liṅga            48%     47%        Not order-dependent (as expected)
  upasarga         63%     62%        Not order-dependent (as expected)

If agreement features (vacana, puruṣa) maintain most of their 
accuracy under shuffling, it's strong evidence that grammatical 
constraints — not word order — drive the signal.
```

### §7.2 — Nuance the theoretical conclusion

The current text says:

> *"Computational validation of grammatical independence from semantics supports the Pāṇinian view..."*

Add a qualification:

> *"Our results demonstrate that next-step grammatical prediction achieves meaningful accuracy, but the contribution of word-order statistics versus grammatical constraints requires further disentanglement. The permutation ablation (§6.2) provides initial evidence, but definitive separation requires testing on verse-order or intentionally order-diverse corpora."*

---

## Room for improvement

### Experiments to add (in order of feasibility)

1. **Permutation test** (easy, high value) — The single best experiment to address this risk

2. **Position encoding ablation** (easy) — Train one model with positional encoding (positions 1, 2, 3... are available to the model) and one without. If removing position information has little effect, word-order is not the primary signal.

3. **Cross-corpus validation** (medium) — Test on a different Sanskrit text with different word-order tendencies. If accuracy transfers, the model learned generalizable grammar, not corpus-specific ordering.

4. **Verse-order training** (medium) — Train on the original verse-order Gītā (before anvaya). If accuracy on agreement features is similar, word order doesn't matter.

5. **Synthetic data with controlled scrambling** (harder but definitive) — Generate synthetic sentences with known grammatical structures in all possible word orders. Train and test across different orderings.

### How to discuss this in the paper

Frame it not as a weakness but as a research question:

> *"How much of the predictive signal comes from grammatical constraints versus word-order regularities? We investigate this with a permutation ablation..."*

This shows intellectual honesty and makes the paper stronger for reviewers who will inevitably ask this question.
