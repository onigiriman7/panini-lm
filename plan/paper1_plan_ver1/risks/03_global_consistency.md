# Risk 03: Local Prediction Cannot Enforce Global Grammatical Consistency

## Severity: MEDIUM

## Which part of the paper this affects

### §4.2–4.3 — Model Architecture and Loss

The model predicts one position at a time:

> *"given the grammatical sequence $(g_1, g_2, \ldots, g_t)$, predict $g_{t+1}$"*

Each prediction is made independently. There is no mechanism to ensure that the *sequence* of predictions forms a globally consistent grammatical structure.

### §6.3 — Hypothesis

The paper hypothesises that certain features should be "highly predictable." But predictability per-position does not guarantee that a full generated sequence (if we chained predictions one after another) would be grammatically valid.

---

## What the risk actually is

### The difference between local accuracy and global consistency

Imagine predicting the weather for each day of the week independently with 80% accuracy. You might predict "sunny" for Monday, "rainy" for Tuesday, "snow" for Wednesday, "40°C heat" for Thursday. Each prediction might be individually plausible, but the sequence as a whole is meteorologically absurd — snow followed by extreme heat doesn't happen.

Similarly, our model predicts each position's grammar independently. Even if each prediction is locally reasonable, the full sequence might violate grammatical constraints that span the entire sentence:

```
Position 1: predicts subanta, prathamā, ekavacana     ✓ (plausible noun subject)
Position 2: predicts subanta, dvitīyā, bahuvacana     ✓ (plausible noun object)
Position 3: predicts tiṅanta, prathama, bahuvacana    ✗ (verb should agree with 
                                                          subject's ekavacana, not 
                                                          object's bahuvacana!)
```

Position 3's prediction might be plausible in isolation (verbs can be bahuvacana), but it violates the agreement constraint with position 1. The model predicts greedily — it doesn't enforce consistency with its own prior predictions.

### Sanskrit grammar as a constraint satisfaction problem

The risk analysis passage identifies this precisely:

> *"Sanskrit grammar behaves more like a constraint satisfaction system than a sequential generation process. Valid sentences = configurations satisfying grammatical rules."*

In a constraint satisfaction problem (CSP), you don't solve it by fixing one variable at a time left-to-right — you need to consider all variables together and find an assignment that satisfies all constraints simultaneously.

**Pāṇinian constraints that span the full sentence:**

| Constraint | Scope | Why local prediction can't enforce it |
|---|---|---|
| Subject-verb vacana agreement | Subject ↔ Verb (anywhere in sentence) | When predicting early words, the verb isn't generated yet |
| Subject-verb puruṣa agreement | Subject ↔ Verb | Same |
| Adjective-noun vibhakti/vacana/liṅga agreement | Adjective ↔ Noun (adjacent or non-adjacent) | Must match a specific noun, but model doesn't track which |
| Exactly one main verb per clause | Entire clause | Per-position prediction can't count verbs |
| Kāraka completeness (certain verbs require certain cases) | Verb ↔ all dependents | A sakarmaka (transitive) verb requires a dvitīyā (accusative) noun somewhere |

### A subtle but important distinction

The paper's goal is not to **generate** grammatically valid sentences — it's to **predict** grammatical features. This distinction matters:

- **Generation**: produce a full valid sequence → global consistency required
- **Prediction**: guess the next position's features → local accuracy sufficient

For the paper's stated purpose (testing whether grammar is predictable), local per-position accuracy is the right metric. The global consistency risk is about **interpretation**: if we claim this model could be used as "stage 1" of a two-stage architecture (§7.1), we need to acknowledge that the predictions aren't guaranteed to be globally consistent.

---

## How serious is this?

### Arguments that this IS serious

1. **The two-stage architecture claim (§7.1) is affected.** If stage 1 (grammar prediction) produces individually plausible but mutually inconsistent grammatical vectors, stage 2 (word selection) might search for a word that matches an impossible grammatical specification.

2. **Evaluation might be misleadingly optimistic.** If 80% of positions are predicted correctly but the 20% that are wrong happen to violate agreement constraints, the resulting sequence might be 0% grammatically valid at the sentence level.

3. **It weakens the Pāṇinian claim (§7.2).** Pāṇini's grammar is all about global consistency — the sūtras jointly determine valid forms. Demonstrating local predictability without global consistency is a weaker validation than the paper implies.

### Arguments that this is manageable

1. **The paper is testing predictability, not generation.** The core hypothesis is "can the next grammar be predicted?" — not "can we generate grammatically perfect sentences from scratch." Per-position accuracy directly tests this.

2. **Agreement partially handles consistency.** If the model learns that verbs agree with prior subjects (in vacana/puruṣa), and subjects precede verbs in most of our data, then the model's predictions will be globally consistent for these features — not by design, but as a consequence of learning the right local patterns.

3. **Global consistency is explicitly not the claim.** The paper doesn't claim to produce valid sentences. Adding a sentence-level validity metric would be interesting but isn't required for the hypothesis test.

---

## Specific paper sections to strengthen

### §4.2 — Clarify that per-position prediction is the intended scope

> *"The model predicts each grammatical position independently given prior context. It does not enforce global sentence-level consistency across predicted positions, which would require a constraint propagation or iterative refinement mechanism. Global grammatical consistency is not required for our hypothesis test, which asks only whether individual positions are predictable."*

### §6.1 — Add a sentence-level validity metric (optional but valuable)

Beyond per-feature accuracy, report:

```
Sentence-level metrics:
  Agreement consistency rate:
    % of predicted sentences where subject-verb agreement 
    is internally consistent across predicted positions
  
  Feature compatibility rate:
    % of predicted positions where the combination of features 
    is a valid combination (e.g., not predicting lakāra for a subanta)
```

This turns a weakness into a measurable result.

### §7.1 — Two-stage architecture needs a consistency caveat

The practical implications section claims that grammar prediction can "dramatically reduce the search space." This is true even without global consistency — each position's search space is reduced independently. But add:

> *"In a deployment setting, a consistency enforcement step (e.g., beam search with grammatical constraints, or iterative refinement) would be needed to ensure that the sequence of predicted grammatical forms is mutually compatible."*

---

## Room for improvement

### Within paper1

1. **Measure agreement consistency** on predicted sequences — easy to compute, turns risk into data
2. **Add the caveat to §7.1** — reviewers will expect this
3. **Frame global consistency as a natural extension** — sets up a clear follow-up paper

### Beyond paper1

The risk analysis passage proposes two approaches that directly address global consistency:

**Approach A: Iterative refinement (diffusion-like)**

```
Step 0: Predict all positions independently (paper1's approach)
Step 1: Check for agreement violations
Step 2: Re-predict violated positions given the predictions of other positions
Step 3: Repeat until convergence (or max iterations)
```

This is a lightweight post-processing step that paper1's model could support.

**Approach B: Constraint-based joint prediction**

```
Instead of P(g_{t+1} | g_1, ..., g_t):
  Model P(g_1, ..., g_n | sentence length n) as a joint distribution
  Use kāraka structure as a constraint graph
  Find the highest-probability assignment that satisfies all constraints
```

This is a full reformulation — the graph/diffusion approaches from the risk analysis — and belongs in a future paper.

### Practical recommendation for paper1

Don't try to fix this — it would change the paper fundamentally. Instead:
1. Acknowledge it clearly (2-3 sentences in §4 or §7)
2. Measure it (agreement consistency rate)
3. Position it as the motivation for future work on constraint-based or iterative approaches
