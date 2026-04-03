# Experiment Design: Step-by-Step Validation Plan

## Overview

This document specifies exactly what experiments to run, what to measure, and how to interpret the results to validate (or invalidate) each hypothesis.

---

## Experiment 1: Baseline Establishment

### Purpose
Before we test our model, we need to know how well a *dumb* model does, so we can measure actual improvement.

### Baselines to compute

**Majority-class baseline**: For each feature, always predict the most common value.

From our data (8,324 tokens across 7 chapters):

| Feature | Most common value | Expected baseline |
|---|---|---|
| Type | subanta (5,409/8,324) | ~65% |
| Vacana | ekavacana (dominant) | ~estimate from data |
| Puruṣa | none (most words are nouns) | ~estimate from data |
| Vibhakti | none (verbs don't have it) | ~estimate from data |

**Bigram baseline**: Predict the next feature value based only on the immediately preceding feature value. This tests whether simple adjacent patterns explain most of the structure, or whether longer-range attention is needed.

**Random baseline**: Predict uniformly at random from valid values. This is the absolute floor.

### How to compute

```python
# For each feature dimension, count value frequencies
# Majority baseline = max frequency / total count
# Bigram: P(feature[t+1] | feature[t]) estimated from training data
```

### What to report

A table with: Feature | Random | Majority | Bigram | Our Model

---

## Experiment 2: Full Model Training and Evaluation

### Purpose
Train the attention-based model and measure per-feature accuracy.

### Training setup

**Data split** (from 697 sentences):
- Train: 70% (~488 sentences, ~5,827 tokens)
- Validation: 15% (~105 sentences, ~1,249 tokens)
- Test: 15% (~105 sentences, ~1,249 tokens)

Split by chapter to avoid leaking stylistic patterns:
- Train: Chapters 1-5 (208 sentences, ~2,858 tokens)
- Validation: Chapter 6 (47 sentences, ~549 tokens)
- Test: Chapter 7 (442 sentences, ~4,917 tokens)

**Note**: Chapter 7 is disproportionately large. Consider alternative splits:
- Random sentence-level split (with seed for reproducibility)
- 5-fold cross-validation across all chapters

**Hyperparameters**:

| Parameter | Value | Rationale |
|---|---|---|
| Attention heads | 4 | Small enough to avoid overfitting, enough to capture multiple relationship types |
| Attention dim | 64 | Sufficient for 54-dim input with some expansion |
| MLP hidden | 128 | 2× attention dim, standard ratio |
| Learning rate | 1e-3 | Standard Adam default |
| Batch size | 32 | ~15 batches per epoch, sufficient gradient signal |
| Epochs | 100-200 | With early stopping on validation loss |
| Dropout | 0.1 | Light regularisation for small dataset |
| Context window | 8 | Start here, ablate later |

### Evaluation metrics

For each of the 9 features:
1. **Accuracy**: % of correctly predicted values on test set
2. **Accuracy (conditional on type)**: Accuracy only for tokens where the feature is applicable
   - e.g., lakāra accuracy computed only when the ground truth type is tiṅanta
3. **Confusion matrix**: To understand what the model confuses
4. **Lift over baseline**: (Model accuracy - Majority baseline) / (1 - Majority baseline)

### Expected results (hypothesis-driven)

| Feature | Baseline (est.) | Expected model | Confidence |
|---|---|---|---|
| Type | ~65% | 75-85% | High |
| Vacana | ~60% | 70-80% | High (agreement signal) |
| Puruṣa | ~75% | 80-90% | High (kartā-kriyā agreement) |
| Vibhakti | ~35% | 45-55% | Moderate |
| Lakāra | ~50% | 55-65% | Moderate |
| Prayoga | ~70% | 72-78% | Low (mostly kartari) |
| Pada | ~60% | 62-68% | Low (weakly constrained) |
| Liṅga | ~45% | 47-52% | Low (semantic feature) |
| Upasarga | ~60% | 62-65% | Low (semantic feature) |

---

## Experiment 3: Ablation — No Attention (MLP Only)

### Purpose
Test whether attention (the ability to look back at specific positions) actually helps, or whether simple local patterns are sufficient.

### Setup
Remove the attention layer. Feed the grammatical features of the immediately preceding position(s) directly into the MLP.

### What this tests
- If MLP-only performs nearly as well as attention → agreement constraints are mostly captured by local (adjacent) patterns
- If attention significantly outperforms MLP → long-range grammatical dependencies are real and important

### Expected result
Attention should help, especially for:
- Puruṣa/vacana of verbs (the agreeing subject may be several positions back)
- Vibhakti of adjectives (the governing noun may not be immediately adjacent)

---

## Experiment 4: Ablation — Context Window Size

### Purpose
How far back does the model need to look for grammatical prediction?

### Setup
Train separate models with context windows of 1, 2, 4, 8, 16 (or full sentence).

### What this tests
- Window 1: Only the immediately preceding word's grammar → purely local patterns
- Window 2-4: Short-range dependencies → local agreement patterns
- Window 8+: Long-range dependencies → clause-level structure

### Expected result
- Sharp improvement from window 1 → 4 (as agreement patterns become visible)
- Diminishing returns beyond window 8 (most Sanskrit sentences are 10-15 tokens)
- Possible slight improvement at full-sentence for long sentences

### Interpretation

```
If window 1 ≈ bigger windows:
  → Grammar is locally predictable, no long-range structure

If window 4 >> window 1, and window 8 ≈ window 4:
  → Medium-range agreement is the key signal (expected result)

If full-sentence >> window 8:
  → Very long-range dependencies exist (unexpected for most Sanskrit)
```

---

## Experiment 5: Ablation — Feature-by-Feature Removal

### Purpose
Which input features are most useful for predicting which output features? This reveals the internal structure of grammatical dependencies.

### Setup
For each input feature, train a model with that feature removed (masked to zeros) and measure the drop in accuracy for each output feature.

### What this tests
- If removing input vacana hurts output puruṣa → number agreement is a real learned dependency
- If removing input vibhakti hurts output puruṣa → the model learned that subject case predicts verb person
- If removing input type hurts everything → type is the most informative signal

### Expected dependency structure

```
Input feature removed    → Output features most affected
─────────────────────      ─────────────────────────────
vacana                   → vacana (of next word), puruṣa
vibhakti                 → type (noun→verb transition signal)
type                     → all features (most informative)
liṅga                    → liṅga (adjective agreement), little else
upasarga                 → nothing much (semantic feature)
```

---

## Experiment 6: Oracle Type Prediction

### Purpose
If we tell the model what TYPE the next word is (noun vs. verb vs. indeclinable), how much better does it predict the specific features?

### Setup
Two conditions:
1. **Standard**: Model predicts type and all other features together
2. **Oracle**: Model is given the correct type; only predicts remaining features

### What this tests
Type determines which features are relevant (verbs have lakāra but not vibhakti; nouns have vibhakti but not lakāra). If oracle significantly boosts accuracy, it means type prediction is the hardest step and the remaining features are well-constrained once type is known.

### Expected result
Oracle should substantially improve accuracy on conditional features:
- Given type = tiṅanta: lakāra, puruṣa, prayoga, pada accuracy should jump
- Given type = subanta: vibhakti, liṅga accuracy should jump

This motivates the two-stage architecture: predict type first, then predict remaining features conditioned on type.

---

## Experiment 7: Practical Search Reduction Measurement

### Purpose
Quantify the actual search space reduction that grammatical prediction enables.

### Setup
For each test token:
1. Note the model's predicted grammatical form
2. Count how many dhātus/prātipadikas in our database are compatible with that form
3. Compare to the total dictionary size

### Metric

$$\text{Reduction factor} = \frac{\text{Total entries in dictionary}}{\text{Entries compatible with predicted form}}$$

### What this tests
Whether the practical claim about search space reduction (roughly 10×) holds.

### Expected measurements

```
For tiṅanta predictions:
  Total dhātus in database:           2,259
  Dhātus compatible with specific form: ~200-500
  Reduction factor:                    ~5-10×

For subanta predictions:
  Total prātipadikas in Gītā vocab:   ~500
  Compatible with specific form:       ~50-100
  Reduction factor:                    ~5-10×
```

---

## Statistical Significance

### Why this matters
With a small dataset (~700 sentences), we need to be careful about claiming results are real and not due to random chance.

### Methods
1. **Bootstrap confidence intervals**: Resample test predictions 1,000 times, compute accuracy each time, report 95% CI
2. **McNemar's test**: Compare model vs. baseline on each example (paired test)
3. **Cross-validation**: 5-fold CV across chapters provides 5 independent accuracy estimates

### Reporting standard
A result is considered meaningful if:
- Model accuracy is above baseline with 95% bootstrap CI not overlapping baseline
- The improvement is consistent across CV folds

---

## Run Order and Dependencies

```
Phase 1: Data analysis (no model needed)
  ├── Compute feature distributions from training data
  ├── Compute majority-class baselines
  ├── Compute bigram baselines
  └── Report data statistics (fills in Section 5.3 of paper)

Phase 2: Core model (main result)
  ├── Implement model architecture
  ├── Train with context window = 8
  ├── Evaluate on test set
  └── Report per-feature accuracy vs. baselines

Phase 3: Ablations (understanding results)
  ├── Exp 3: MLP-only ablation
  ├── Exp 4: Context window ablation
  ├── Exp 5: Feature removal ablation
  └── Exp 6: Oracle type experiment

Phase 4: Practical implications
  └── Exp 7: Search reduction measurement

Phase 5: Statistical validation
  ├── Bootstrap CIs for all reported numbers
  └── Cross-validation consistency check
```

Each phase depends on the previous one. Phases 3-5 are optional for the core paper but strongly recommended.
