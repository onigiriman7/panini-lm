# Risk 01: Sequential Bias in Free Word-Order Sanskrit

## Severity: HIGH

## Which part of the paper this affects

### §3 — Task Formulation (lines 73-79 of paper1.md)

The paper defines the task as:

> *"The task is: given the grammatical sequence $(g_1, g_2, \ldots, g_t)$, predict $g_{t+1}$."*

This formulation treats Sanskrit text as a **left-to-right sequence** where position $t+1$ comes "after" positions $1$ through $t$. The model sees grammar in the order words appear in the text and predicts the next one.

### §4.2 — Architecture

The attention mechanism operates over positions $1$ through $t$ to predict position $t+1$. This is **causal** (left-to-right) attention — the model only looks backward.

---

## What the risk actually is

### Sanskrit word order is grammatically free

Unlike English, where "The dog bites the man" and "The man bites the dog" have different meanings because word order determines who is the subject, Sanskrit uses **case endings** (vibhakti) to mark grammatical roles. Word order is stylistic, not grammatical:

```
All of these mean "Rāma sees Sītā":

  rāmaḥ sītām paśyati          (Subject-Object-Verb)
  sītām rāmaḥ paśyati          (Object-Subject-Verb)
  paśyati rāmaḥ sītām          (Verb-Subject-Object)
  rāmaḥ paśyati sītām          (Subject-Verb-Object)
  sītām paśyati rāmaḥ          (Object-Verb-Subject)
  paśyati sītām rāmaḥ          (Verb-Object-Subject)
```

All six orderings are **grammatically identical**. The vibhakti markings (rāmaḥ = prathamā/subject, sītām = dvitīyā/object) carry the syntactic information, not the position.

### Why this is a problem for next-step prediction

If the same grammatical content can appear in any order, then the "next" grammatical form is not uniquely determined by position. The model's prediction target depends on which word-order permutation the author happened to choose — a stylistic decision, not a grammatical one.

**Concrete example**: Given the grammar sequence `[subanta-prathamā-eka, subanta-dvitīyā-eka]`, what comes next?

- In SOV order: the verb → `tiṅanta`
- In SVO order: already saw the verb, so another noun or sentence boundary
- In OSV order: the subject noun → `subanta-prathamā`

The "correct" answer depends on the surface order, which is arbitrary.

### What the model might actually learn

Instead of learning "grammar constrains grammar" (the intended hypothesis), the model might learn:

> "In the Bhagavad Gītā's prose version, SOV order is most common, so after two nouns predict a verb."

This is a **word-order statistic**, not a grammatical constraint. It would produce above-baseline accuracy but for the wrong reason.

---

## How serious is this?

### Arguments that this IS serious

1. **The hypothesis is about grammar, not word order.** If the model succeeds by learning word-order statistics, the result doesn't validate the Pāṇinian claim about grammatical independence.

2. **Single-corpus bias.** The Gītā's prose form has been manually rewritten into a relatively standard word order (close to anvaya/prose order). This means the word order in our data is even more regular than typical Sanskrit — making word-order statistics even more reliable as a shortcut.

3. **Reviewers will flag this.** Any NLP reviewer familiar with free word-order languages will immediately ask: "How do you distinguish grammatical signal from word-order signal?"

### Arguments that this is manageable

1. **Even word-order statistics reflect grammatical constraints.** The reason SOV is common in Sanskrit is that grammatical roles drive typical ordering patterns. The signal isn't purely arbitrary.

2. **Agreement features transcend word order.** Regardless of whether the subject comes before or after the verb, the verb must agree with it in vacana and puruṣa. If the model achieves high accuracy on these features, it's capturing agreement — not just position.

3. **The paper's ablation on context window (§6.2, ablation 2) partially addresses this.** If the model performs well with window size 1 (only the immediately preceding word), it's learning local ordering patterns. If it needs larger windows, it's learning longer-range constraints.

---

## Specific paper sections to strengthen

### §3 — Add explicit acknowledgment

The task formulation should acknowledge that Sanskrit has free word order, and explain why next-step prediction is still a valid first test:

> *"Sanskrit permits flexible word order, meaning the sequence $(g_1, \ldots, g_t)$ reflects one of potentially many valid orderings. Our model therefore targets a weaker claim than full grammatical determination: that the typical ordering conventions of prose Sanskrit, combined with agreement constraints, produce sufficient sequential structure for next-step prediction. We test whether order-independent agreement constraints (e.g., kartā-kriyā vacana agreement) are a substantial component of this signal in ablation 3."*

### §6.2 — Add a word-order ablation

**New ablation proposal**: Randomly permute the word order within each sentence (preserving all grammatical annotations) and retrain. If accuracy drops sharply, the model was relying on word-order statistics. If accuracy on agreement features (vacana, puruṣa) remains high even under permutation, the model has learned genuine grammatical constraints.

```
Ablation 4: Word-order permutation
  ├── Randomly shuffle word positions within each sentence
  ├── Keep all grammatical features unchanged
  ├── Retrain model on permuted data
  └── Compare per-feature accuracy to unpermuted model

Expected results:
  - Type prediction:     ↓ sharply (type sequence is order-dependent)
  - Vacana/puruṣa:       ↓ moderately (agreement is order-independent, 
                              but harder to find without positional cues)
  - Vibhakti:            ↓ moderately
  - Liṅga/upasarga:      ↔ unchanged (these were never order-dependent)
```

### §7 — Frame as a limitation and future direction

The implications section should explicitly acknowledge this limitation and point toward order-independent formulations as future work.

---

## Room for improvement: Beyond sequential prediction

### What would a better formulation look like?

The risk analysis passage suggests several alternatives. For paper1, the most practical improvement is not to abandon sequential prediction (which would require a different paper), but to:

1. **Add the permutation ablation** described above to quantify how much of the signal is order-dependent
2. **Report agreement-conditional accuracy** — accuracy on vacana/puruṣa computed only for verb positions where the subject is in the context window, regardless of whether it's adjacent
3. **Discuss the limitation honestly** — frame sequential prediction as a pragmatic first step and point toward constraint-based models as future work

### What future papers could do

| Approach | How it addresses the risk | Complexity |
|---|---|---|
| Permuted training | Tests robustness to word order | Low (easy to implement) |
| Bidirectional attention (BERT-style masking) | Removes left-to-right assumption | Medium |
| Graph neural network on kāraka relations | Models grammar as a constraint graph, no sequence assumption | High |
| Diffusion-based refinement | Starts from random, converges to valid grammar configurations | Very high |

Paper1 should implement the first, acknowledge the second, and point to the third and fourth as future work.
