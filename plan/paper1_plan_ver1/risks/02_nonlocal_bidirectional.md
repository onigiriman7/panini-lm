# Risk 02: Non-Local and Bidirectional Dependencies

## Severity: HIGH

## Which part of the paper this affects

### §2.1 — Pāṇinian Grammatical Features

The paper correctly identifies the key agreement constraints:

> *"a verb must agree with its kartā in puruṣa and vacana; a viśeṣaṇa must agree with its viśeṣya in vibhakti, vacana, and liṅga"*

These agreement relations are **bidirectional** and **non-local**:

- **Bidirectional**: The verb constrains the subject just as much as the subject constrains the verb. In a grammatically complete sentence, both must agree — neither "causes" the other.
- **Non-local**: The agreeing words can be arbitrarily far apart. In long sentences, the subject and verb might be separated by many intervening words (relative clauses, appositional phrases, etc.).

### §4.2 — Architecture (Attention Layer)

The paper uses **causal self-attention** — position $t$ can only attend to positions $1$ through $t$. This means:

```
Position:    1     2     3     4     5     6     7     8
Word:        rāmaḥ sītāyāḥ sundaraṃ putraṃ   ...    ...    ...   paśyati
             ──────────────────────────────────────────────────► (can see)
                                                                 ◄────── (cannot see)
```

Position 8 (the verb) can look back and see position 1 (the subject) — so the verb's agreement with the subject is learnable. But position 1 (the subject) **cannot look ahead** to see position 8 (the verb).

---

## What the risk actually is

### Agreement is symmetric, but the model sees it asymmetrically

In Pāṇinian grammar, the relation between kartā (agent) and kriyā (action) is a **symmetric constraint**: given one, the other is constrained. But in left-to-right prediction:

- When predicting the **verb** (which comes later in SOV order): the model has already seen the subject, so it can use agreement → **works**
- When predicting the **subject** (which comes first in SOV order): the model hasn't seen the verb yet, so it cannot use agreement → **doesn't work**

This means the model can learn **"given subject, predict verb agreement"** but not **"given verb, predict subject agreement"** — which is an artificial asymmetry imposed by the sequential formulation.

### Real-world consequences

**Example where this fails**:

Consider a sentence in OVS order (which is valid in Sanskrit):

```
sītām             paśyati          rāmaḥ
(object, dvitīyā) (verb, prathama) (subject, prathamā)
  position 1         position 2       position 3
```

When predicting position 2 (the verb), the model has seen only position 1 (the object). The object is in dvitīyā — it's NOT the subject. The verb's puruṣa and vacana must agree with the subject (rāmaḥ, position 3), which the model **hasn't seen yet**.

The model must predict the verb's agreement features without having seen the thing it agrees with. This is fundamentally impossible from grammatical context alone — the model would need to fall back on statistical defaults.

### How often does this happen in our data?

The Gītā's prose form is mostly in **anvaya (natural prose) order**, which is close to SOV. This means:

- Most subjects come **before** their verbs → agreement IS visible
- But not all — especially in verse-derived prose, emphatic constructions, and quotations

Estimated impact: 10-25% of subject-verb agreement instances may have the subject after the verb, making agreement features unpredictable for those instances.

---

## How serious is this?

### Arguments that this IS serious

1. **It artificially limits performance on agreement features.** The paper hypothesises high accuracy on vacana and puruṣa, but this is only achievable for the majority of cases where agreement is left-to-right. The overall accuracy will be dragged down by cases where agreement is right-to-left.

2. **It could mask a stronger result.** If the true grammatical signal is even stronger than what our model captures (because some agreement is backward-looking), a bidirectional model would reveal this. Our sequential model underestimates the real predictive power of grammar.

3. **Non-local dependencies span variable distances.** Even when the subject comes before the verb, many words might intervene. The attention mechanism can theoretically bridge any distance within the context window, but in practice, a single attention layer with limited capacity may struggle with long-distance dependencies.

### Arguments that this is manageable

1. **SOV dominates in our data.** Since the Gītā is in prose order, most subject-verb pairs have the subject first. The model should still capture the majority of agreement patterns.

2. **The paper doesn't claim perfection.** The hypothesis is that grammar is *partially* predictable — not that every feature is perfectly determined. Some information loss from directionality is expected and acceptable.

3. **Causal attention is standard technology.** GPT and all autoregressive models face the same limitation. The NLP community accepts this as a trade-off for practical generation capability.

---

## Specific paper sections to strengthen

### §2.1 — Acknowledge the directionality issue

Add a sentence clarifying that agreement in Pāṇinian grammar is bidirectional, but the model can only exploit it in one direction:

> *"These agreement constraints are symmetric — each constrains the other — but our left-to-right prediction formulation can only exploit them when the constraining element precedes the constrained one in surface order."*

### §4.2 — Discuss the choice of causal vs. bidirectional attention

The architecture section should briefly justify why causal attention was chosen despite the bidirectionality limitation:

> *"We use causal self-attention because the prediction task is autoregressive (predicting position $t+1$ from prior positions). A bidirectional formulation (e.g., masked language modeling) would allow exploiting backward agreement but changes the task from next-step prediction to fill-in-the-blank, which we leave to future work."*

### §6.2 — Add agreement direction analysis

**New analysis proposal**: For each verb in the test set, check whether its agreeing subject appears before or after it. Report accuracy separately:

```
Agreement direction analysis:

Verb with subject BEFORE it in text:
  vacana accuracy:  __%
  puruṣa accuracy:  __%

Verb with subject AFTER it in text:
  vacana accuracy:  __%
  puruṣa accuracy:  __%
```

If the first accuracy is much higher than the second, it confirms that the model is using directional agreement — which supports the hypothesis (agreement is exploitable) but reveals the limitation (only in one direction).

---

## Room for improvement

### Short-term (within paper1)

1. **Report directional accuracy** as described above — this turns a weakness into an insight
2. **Acknowledge the limitation** explicitly in §3 and §7
3. **Frame bidirectional approaches as future work** — this actually strengthens the paper by showing awareness and pointing to a clear extension

### Medium-term (follow-up work)

| Approach | What it does | Feasibility |
|---|---|---|
| BERT-style masked prediction | Mask one position, predict from both sides | Medium — changes the task but keeps the same feature representation |
| Bidirectional attention + masking | During training, allow looking at all positions except the target | Medium — straightforward modification |
| Two-pass model | First pass left-to-right, second pass right-to-left, combine | Medium — doubles model size but conceptually simple |

### Long-term (different paradigm)

The most principled solution is to abandon sequential prediction entirely and model grammar as a **constraint graph** where all positions are simultaneously visible and the task is to determine compatibility — not prediction. This is the graph/constraint-based approach from the risk analysis passage:

```
Input:  Partial sentence with some grammar features filled in
Task:   Determine which feature assignments for remaining positions 
        are grammatically valid
Method: Graph neural network over kāraka dependency structure
```

This removes all directional bias and matches the Pāṇinian formulation most faithfully, but is a substantially different paper.
