# Risk 05: Alternative Architectures May Better Match the Problem Structure

## Severity: MEDIUM

## Which part of the paper this affects

### §4 — Model (entire section)

The paper proposes a specific architecture: single-layer causal multi-head attention → MLP → per-feature softmax heads. The risk analysis passage argues that this architecture **does not match the structure of the problem** and proposes four alternatives.

This document evaluates each alternative — what it offers, what it costs, and whether paper1 should adopt it, acknowledge it, or save it for future work.

---

## The Mismatch Problem

The paper correctly identifies that Sanskrit grammar involves:
- Agreement constraints across variable distances
- Mutual (bidirectional) dependencies between words
- Sentence-level validity requirements

But the chosen architecture is:
- Sequential (left-to-right)
- Unidirectional (causal attention)
- Position-level (predicts one token, no global enforcement)

This is like using a ruler to measure a curve — it works locally but misses the true shape. The question is whether the mismatch is serious enough to invalidate the results, or an acceptable approximation for a first paper.

---

## Alternative 1: Sequential Aggregation (the paper's current approach)

### What it is
This is essentially what paper1 proposes: encode each word as a grammar vector, use attention to aggregate past context, predict the next grammar vector.

### Strengths
- Simple, well-understood architecture
- Easy to implement and train
- Standard evaluation metrics apply (accuracy per feature)
- Directly comparable to LM-style next-token prediction literature
- Sufficient to test the core hypothesis: "is grammar predictable from grammar?"

### Weaknesses (the risks)
- Left-to-right bias (Risk 01)
- Cannot use backward agreement (Risk 02)
- No global consistency (Risk 03)
- May learn word-order statistics (Risk 04)

### Verdict for paper1
**Keep as the primary approach.** It's the simplest test of the core hypothesis. The weaknesses should be acknowledged, not fixed — fixing them would produce a different paper.

---

## Alternative 2: Multi-Token Prediction

### What it is
Instead of predicting only the next token's grammar, predict the grammar of the next $k$ tokens simultaneously.

```
Standard:     (g_1, ..., g_t) → g_{t+1}
Multi-token:  (g_1, ..., g_t) → (g_{t+1}, g_{t+2}, ..., g_{t+k})
```

### How it maps to the paper

This would modify §3 (task formulation) and §4.2 (architecture — the output heads would predict $k$ sets of features instead of 1).

### What it offers
- **Captures short-range dependencies between predicted positions.** If position $t+1$ is a noun and position $t+2$ is its adjective, multi-token prediction can enforce their agreement because both are predicted together.
- **More efficient inference.** Predicting 4 tokens at once is 4× faster than sequential generation.

### What it costs
- More complex output structure ($k$ × 9 output heads instead of 9)
- Requires deciding on $k$ in advance
- For variable-length sentences, what happens when $t + k$ exceeds sentence length?

### Assessment for paper1

**Worth mentioning as a straightforward extension but not worth implementing in paper1.**

The paper's hypothesis is about whether grammar is predictable at all — the simplest formulation (predict 1 token) is the cleanest test. Multi-token prediction is an engineering improvement, not a conceptual one.

**Suggested addition to §7 (future work):**
> *"Multi-token prediction, where the model jointly predicts the grammatical forms of the next $k$ positions, could capture short-range co-constraints between adjacent predictions and warrants future investigation."*

---

## Alternative 3: Graph / Constraint-Based Model

### What it is
Represent each sentence as a graph:
- **Nodes**: word positions, each labeled with grammatical features
- **Edges**: grammatical relationships (kāraka relations, agreement links)

A graph neural network (GNN) learns valid configurations by message-passing between connected nodes.

```
         kartā-kriyā
  rāmaḥ ──────────── gacchati
  (sub, 1st, eka)    (tiṅ, 3rd, eka)
     │                    │
     └── viśeṣaṇa ──── sundaraḥ
                       (sub, 1st, eka, masc)
```

### How it maps to the paper

This would fundamentally replace §3 (task formulation — no longer next-step prediction) and §4 (entirely different architecture). It would also require different training data: the current `gita_training.json` includes `adjacency_edges` with `link_type` labels like `"kartā-kriyā"`, which is exactly the graph structure this approach needs.

### What it offers
- **No word-order dependence.** The graph operates on grammatical relations, not positions. Perfectly addresses Risk 01 and Risk 04.
- **Bidirectional constraints.** Message-passing goes both ways along edges. Addresses Risk 02.
- **Global consistency.** The GNN refines all nodes jointly. Partially addresses Risk 03.
- **Matches Pāṇinian theory most closely.** Pāṇini's grammar is essentially a set of constraints on grammatical relations, not a sequential generation process.

### What it costs
- **Requires dependency annotation.** Need kāraka/dependency edges, not just feature sequences. Our data has these (`adjacency_edges` in `gita_training.json`) but they may not cover all sentences.
- **Different task formulation.** Not next-step prediction — more like constraint completion or graph validity. Harder to compare with standard NLP baselines.
- **More complex implementation.** GNNs are harder to train, tune, and interpret than Transformer-style attention.
- **Needs parsed input.** In a real application, you'd need to parse the sentence to get the graph — a chicken-and-egg problem (you need grammar to build the graph, but the graph is how you predict grammar).

### Assessment for paper1

**Not appropriate for paper1, but the strongest candidate for paper2.**

Paper1's value is in defining the hypothesis and showing initial evidence. A graph-based model is the principled way to test it without the confounds of sequential prediction, but it's a different paper with different methods.

**Suggested addition to §7.2:**
> *"A natural extension is to model grammatical structure as a constraint graph over kāraka relations, using graph neural networks to learn valid feature configurations without any sequential assumption. This formulation would provide the strongest test of the Pāṇinian hypothesis, as it mirrors the relational structure of the Aṣṭādhyāyī directly."*

**Our data already supports this.** The `adjacency_edges` field with `link_type` labels (kartā-kriyā etc.) in `gita_training.json` means the graph structure is available — the data pipeline work won't need to be redone.

---

## Alternative 4: Diffusion-Based Model

### What it is
Start with a "noisy" sentence: random grammar assignments for each position. Iteratively refine the assignments to reduce grammatical violations, converging toward a valid configuration.

```
Step 0:  [random, random, random, random, random]
Step 1:  [sub-pra-eka, random, random, random, random] 
         (type becomes clearer)
Step 2:  [sub-pra-eka, sub-dvi-eka, random, tiṅ-?, random]
         (some nouns and verb type emerge)
Step 3:  [sub-pra-eka, sub-dvi-eka, avyaya, tiṅ-3rd-eka, sub-pra-eka]
         (agreement propagates: verb matches subject)
...
Step N:  [valid grammatical configuration]
```

### How it maps to the paper

This would entirely replace §3 and §4. The task is no longer "predict the next token" — it's "refine a random assignment into a valid one."

### What it offers
- **No directional bias.** All positions are refined simultaneously — no left-to-right assumption. Addresses Risks 01, 02.
- **Global consistency by design.** The refinement process explicitly aims for a globally valid configuration. Addresses Risk 03.
- **Naturally handles free word order.** Position in the sequence is irrelevant; what matters is the final configuration. Addresses Risk 04.
- **Matches the "constraint satisfaction" insight.** The risk analysis passage correctly identifies this as the most principled formulation.

### What it costs
- **Dramatically more complex.** Diffusion models require careful noise scheduling, training is expensive, convergence is not guaranteed.
- **No easy baselines.** Hard to compare against standard NLP approaches.
- **Defining "grammatical violations" to minimise requires a differentiable grammar checker** — this is non-trivial.
- **Overkill for the current hypothesis.** We want to know if grammar is predictable; we don't need the most theoretically pure model to test this.
- **Very few prior examples.** Diffusion for discrete structured data (as opposed to images or continuous signals) is still an active research area with limited proven approaches.

### Assessment for paper1

**Not appropriate for paper1 or even paper2. This is a long-term research direction.**

The theoretical motivation is compelling, but the engineering challenges are substantial. This belongs in a future paper after the basic hypothesis is validated and after graph-based approaches have been explored.

**Suggested addition to §7 (one sentence in future work):**
> *"A diffusion-based approach, in which grammatical features are refined from random initialisation toward valid configurations, is a theoretically attractive direction that naturally handles free word order and enforces global consistency."*

---

## Summary: Which alternative should paper1 adopt?

| Alternative | Adopt in paper1? | Why |
|---|---|---|
| Sequential (current) | ✅ **Keep** | Simplest valid test of the core hypothesis |
| Multi-token | ❌ Mention only | Engineering improvement, not conceptual advance |
| Graph/constraint | ❌ Set up for paper2 | Strongest principled approach, but different paper |
| Diffusion | ❌ Mention briefly | Too speculative and complex for near-term |

### What paper1 should do

1. **Keep the sequential model** as the primary approach
2. **Add the permutation test** (from Risk 01/04) to quantify how much signal is order-independent
3. **Acknowledge risks explicitly** — this shows research maturity, not weakness
4. **Frame alternatives as a roadmap** — from sequential (paper1) to graph-based (paper2) to diffusion (long-term)

### The roadmap this implies

```
Paper 1 (current):
  "Can grammar be predicted from grammar?"
  Method: Sequential next-step prediction
  Result: Yes, partially — agreement features are predictable
  Limitation: Can't separate grammar signal from word-order signal entirely

Paper 2 (next):
  "Is grammar predictable without word-order dependence?"
  Method: Graph neural network on kāraka relations
  Data: Same Gītā corpus, using existing adjacency_edges
  Result: Tests whether the signal survives removal of sequence bias

Paper 3 (future):
  "Can valid grammatical configurations be generated from scratch?"
  Method: Diffusion/iterative refinement
  Result: Tests whether grammar is not just predictable but generatable
```

This roadmap strengthens paper1 because it positions the current work as a necessary first step — not a definitive answer, but a foundation that motivates and enables more powerful future approaches.
