# Risk 06: Consolidated Recommendations for paper1.md

This document distils every risk into specific, actionable changes to the paper — sorted by priority.

---

## PRIORITY 1: Must-Do (risks that reviewers will certainly flag)

### 1A. Add the permutation ablation experiment

**What**: Randomly shuffle word order within each sentence, retrain, compare accuracy.

**Where in paper**: §6.2 (Ablations) — add as Ablation 4

**Why it matters**: This is the single strongest experiment for separating grammatical signal from word-order signal. Without it, a reviewer can dismiss the results as "the model just learned SOV order."

**Addresses**: Risk 01 (sequential bias), Risk 04 (word-order confound)

**Draft text for §6.2**:
```
ablation 4  :  word-order permutation
               randomly shuffle the order of grammatical vectors 
               within each training sentence, retrain the model, 
               and compare per-feature accuracy.
               measures how much of the prediction signal comes 
               from word-order regularities vs. grammatical constraints.
               hypothesis: agreement features (vacana, puruṣa) 
               retain most accuracy; type prediction drops sharply.
```

---

### 1B. Acknowledge free word order in task formulation

**What**: Add 2-3 sentences to §3 explaining that Sanskrit has free word order and what this means for the prediction task.

**Where in paper**: §3 (Task formulation) — after the task definition

**Why it matters**: Any reviewer who knows Sanskrit (or free word-order languages) will immediately question the sequential formulation. Pre-empting this shows awareness.

**Addresses**: Risk 01

**Draft text**:
> Sanskrit permits flexible word order: the grammatical content of a sentence is preserved across permutations of its surface ordering. Our sequential prediction formulation therefore targets the specific ordering conventions of prose Sanskrit, which approximate but do not enforce a fixed word order. The permutation ablation (§6.2) quantifies how much of the prediction signal depends on these ordering conventions versus order-independent grammatical constraints.

---

### 1C. Add a limitations paragraph

**What**: A short paragraph in §7 or §8 that honestly states what the paper does NOT show.

**Where in paper**: New subsection §7.3 or fold into §8

**Why it matters**: Every strong paper has a limitations section. Its absence signals either naivety or evasion.

**Addresses**: All risks

**Draft text**:
> **Limitations.** Our sequential formulation captures only left-to-right grammatical dependencies; backward-looking agreement (where the constraining element follows the constrained one) is not exploitable. Per-position prediction does not enforce global sentence-level consistency. The prose-form Gītā has more regular word order than typical Sanskrit verse, which may inflate accuracy due to positional statistics. Results from a single text require validation on diverse corpora. These limitations motivate future work on bidirectional and graph-based approaches.

---

## PRIORITY 2: Should-Do (significantly strengthens the paper)

### 2A. Report agreement-direction accuracy

**What**: For each verb, check whether its agreeing subject is before or after it. Report vacana/puruṣa accuracy separately for each case.

**Where in paper**: §6.1 or new results subsection

**Why it matters**: Demonstrates that the model captures real agreement (high accuracy when subject precedes) and reveals its limitation (lower accuracy when subject follows). Turns Risk 02 into a measurable insight.

**Addresses**: Risk 02 (non-local bidirectional dependencies)

---

### 2B. Report sentence-level agreement consistency

**What**: After predicting all positions in a test sentence one-by-one, check whether the predicted subject and verb actually agree with each other.

**Where in paper**: §6.1

**Why it matters**: Tests whether per-position predictions are globally coherent. Even a simple metric like "% of sentences where predicted subject and verb agree" is valuable.

**Addresses**: Risk 03 (global consistency)

---

### 2C. Discuss the anvaya/prose-form data explicitly

**What**: Note that the Gītā has been converted to prose order (anvaya), what this means for word-order regularity, and that anvaya itself is evidence that grammar determines word order.

**Where in paper**: §5.1

**Why it matters**: Shows the reviewer you understand the data's properties and have thought about their implications.

**Addresses**: Risk 04 (word-order confound)

---

## PRIORITY 3: Nice-to-Have (adds depth, not strictly necessary)

### 3A. Future work: graph-based model

**What**: 2-3 sentences pointing toward kāraka graph modeling as a natural next step.

**Where in paper**: §7.2 or new §7.3 (Future Work)

**Why it matters**: Shows the paper is a stepping stone in a research program, not an endpoint. The adjacency_edges data in gita_training.json means the groundwork is already laid.

**Addresses**: Risk 05 (alternative architectures)

**Draft text**:
> A natural extension is to model grammatical structure as a constraint graph over kāraka relations, removing the sequential assumption entirely. Our annotated corpus includes kāraka dependency edges (kartā-kriyā, karma-kriyā, etc.) that can serve as the input graph for such a model.

---

### 3B. Future work: diffusion/iterative refinement

**What**: 1-2 sentences mentioning the diffusion approach as a long-term direction.

**Where in paper**: §7.2

**Draft text**:
> A longer-term direction is diffusion-based grammatical refinement, where feature assignments are iteratively refined from random initialisation toward valid configurations, naturally handling free word order and enforcing global consistency.

---

### 3C. Position-encoding ablation

**What**: Train the model with and without positional encoding to measure how much position-specific information (vs. content-specific information) drives accuracy.

**Where in paper**: §6.2 (additional ablation)

**Why it matters**: A cleaner version of the permutation test — if removing position information has little effect, the signal is truly grammatical.

**Addresses**: Risk 04

---

## Implementation Checklist

```
[ ] MUST:  Add permutation ablation to §6.2
[ ] MUST:  Add free word-order acknowledgment to §3
[ ] MUST:  Add limitations paragraph to §7/§8
[ ] SHOULD: Report agreement-direction accuracy in results
[ ] SHOULD: Report sentence-level consistency metric
[ ] SHOULD: Add anvaya discussion to §5.1
[ ] NICE:  Add graph-based model to future work
[ ] NICE:  Add diffusion mention to future work
[ ] NICE:  Add position-encoding ablation
```

---

## What NOT to change

The risk analysis passage proposes alternative architectures (graph models, diffusion models, multi-token prediction). **Paper1 should NOT adopt any of these.** The sequential formulation is the right first step because:

1. It's the simplest valid test of the core hypothesis
2. It's comparable to existing NLP literature (autoregressive LMs)
3. It establishes a baseline that more sophisticated models can improve upon
4. Changing the architecture now would delay the paper significantly

The right strategy is: **keep the simple model, add the right ablations, acknowledge limitations honestly, and point to alternatives as future work.** This is the standard structure of a strong first paper in a new research program.
