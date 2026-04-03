# Risk Analysis Overview

This folder contains a detailed examination of structural risks in the paper's approach, arising from a fundamental tension:

> **Sanskrit grammar is a constraint satisfaction system, but the paper models it as a sequential prediction problem.**

This is not a fatal flaw — but it is a design choice with consequences that must be acknowledged, measured, and defended. Each risk document below maps a specific concern to the exact section of the paper it affects, analyses how serious the risk is, and proposes concrete mitigations.

---

## Risk Index

| # | Risk | Severity | Paper sections affected |
|---|---|---|---|
| 01 | Sequential bias: next-token prediction assumes left-to-right order, but Sanskrit has free word order | **High** | §3 Task formulation, §4.2 Architecture |
| 02 | Non-local and bidirectional dependencies cannot be captured by causal (left-to-right) attention | **High** | §4.2 Attention layer, §2.1 Agreement rules |
| 03 | Local prediction vs. global consistency: predicting one position at a time cannot enforce sentence-level grammatical validity | **Medium** | §4.2-4.3 Model & Loss, §6.3 Hypothesis |
| 04 | The model may learn corpus word-order statistics rather than genuine grammatical constraints | **Medium** | §5.1 Corpus, §6.2 Ablations |
| 05 | Alternative architectures may better match the problem structure | **Medium** | §4 Model (entire section) |

---

## Summary of the Core Tension

The paper's hypothesis is linguistically sound:

> *Grammatical form is predictable from grammatical form alone.*

But the **operationalisation** — next-step sequential prediction — may not be the best way to test this hypothesis. The risk is not that the hypothesis is wrong, but that:

1. A positive result could be attributed to word-order statistics rather than grammatical constraints
2. A negative result might reflect the mismatch between formulation and phenomenon, not a failure of the hypothesis itself
3. Stronger results might be achievable with a formulation that better matches Sanskrit's grammatical structure

Each risk document explores one facet of this tension in detail.
