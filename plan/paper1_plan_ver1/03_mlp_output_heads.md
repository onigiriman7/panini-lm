# Module 3: MLP + Output Heads

## Purpose

Transform the attended representation into per-feature grammatical predictions. This module implements the MLP and output heads described in section 4.2 of the paper. It takes the context-aware representation produced by the attention layer and produces independent probability distributions over each of the 9 grammatical features for the next primitive.

---

## Inputs

| Input | Shape | Source |
|---|---|---|
| $\mathbf{h}_{\text{out}}$ | $(B, d_{\text{model}})$ | Attention module output (representation at last context position) |

---

## Transformation

### Step 1: Two-layer feed-forward network

$$\mathbf{z}_1 = \text{ReLU}(\mathbf{h}_{\text{out}} W_1 + b_1)$$
$$\mathbf{z} = \mathbf{z}_1 W_2 + b_2$$

where:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{hidden}}}$, $b_1 \in \mathbb{R}^{d_{\text{hidden}}}$
- $W_2 \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{mlp\_out}}}$, $b_2 \in \mathbb{R}^{d_{\text{mlp\_out}}}$

### Step 2: Output heads (one per grammatical feature)

Nine independent linear projections followed by softmax, each predicting one grammatical feature of the next primitive:

$$\hat{f}_j = \text{softmax}(\mathbf{z} \, W_j^{\text{out}} + b_j^{\text{out}})$$

| Head | Feature | Output dim | Classes |
|---|---|---|---|
| 1 | primitive type | 3 | dhātu, prātipadika, avyaya |
| 2 | lakāra | 11 | 10 lakāras + NULL |
| 3 | puruṣa | 4 | prathama, madhyama, uttama + NULL |
| 4 | vacana | 4 | eka, dvi, bahu + NULL |
| 5 | prayoga | 4 | kartari, karmaṇi, bhāve + NULL |
| 6 | pada | 3 | parasmaipada, ātmanepada + NULL |
| 7 | vibhakti | 9 | prathama–saptamī, sambodhana + NULL |
| 8 | liṅga | 4 | pul, strī, napuṃsaka + NULL |
| 9 | upasarga | 20 | 18 upasargas + none + NULL |

### Step 3: Loss computation

Per-feature cross-entropy loss with per-feature weights:

$$\mathcal{L} = \sum_{j=1}^{9} \lambda_j \cdot \mathcal{L}_{\text{CE}}(\hat{f}_j, f_j^{*})$$

**NULL masking rule:** When the ground-truth primitive type makes a feature inapplicable (e.g., lakāra for a prātipadika), that feature's loss contribution is zeroed out. The model is not penalised for its prediction on inapplicable features — only the applicable features contribute to the gradient.

---

## Hyperparameters

| Parameter | Symbol | Proposed value |
|---|---|---|
| MLP hidden dim | $d_{\text{hidden}}$ | TBD |
| MLP output dim | $d_{\text{mlp\_out}}$ | TBD (can equal $d_{\text{hidden}}$) |
| Feature weights | $\lambda_j$ | 1.0 for all (uniform), or tuned |
| Dropout | $p$ | TBD |

Total parameter budget target: ~50,000 across the full model (attention + MLP + heads).

---

## Outputs

| Output | Shape | Description |
|---|---|---|
| $\hat{f}_1$ | $(B, 3)$ | Probability distribution over primitive type |
| $\hat{f}_2$ | $(B, 11)$ | Probability distribution over lakāra |
| $\hat{f}_3$ | $(B, 4)$ | Probability distribution over puruṣa |
| $\hat{f}_4$ | $(B, 4)$ | Probability distribution over vacana |
| $\hat{f}_5$ | $(B, 4)$ | Probability distribution over prayoga |
| $\hat{f}_6$ | $(B, 3)$ | Probability distribution over pada |
| $\hat{f}_7$ | $(B, 9)$ | Probability distribution over vibhakti |
| $\hat{f}_8$ | $(B, 4)$ | Probability distribution over liṅga |
| $\hat{f}_9$ | $(B, 20)$ | Probability distribution over upasarga |
| $\mathcal{L}$ | scalar | Total weighted loss |

---

## KPIs / Correctness Metrics

### 1. Shape correctness
- Each output head $\hat{f}_j$ must have shape $(B, |V_j|)$ where $|V_j|$ is the vocabulary size of feature $j$.
- The total number of output dimensions across all heads must equal $3 + 11 + 4 + 4 + 4 + 3 + 9 + 4 + 20 = 62$.

### 2. Probability validity
- Each $\hat{f}_j$ must be a valid probability distribution: all values in $[0, 1]$ and summing to 1 per sample.

### 3. NULL masking correctness
- When ground-truth type is prātipadika, loss contributions from lakāra, puruṣa, prayoga, pada, and upasarga heads must be exactly 0.
- When ground-truth type is avyaya, loss contributions from all heads except primitive type must be exactly 0.
- When ground-truth type is dhātu, loss contributions from vibhakti and liṅga heads must be exactly 0.
- Verify by constructing examples of each type and asserting loss values.

### 4. Per-feature accuracy (primary evaluation metric)

For each feature $j$, report:

| Metric | Definition |
|---|---|
| **Accuracy** | $\frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\arg\max(\hat{f}_j^{(i)}) = f_j^{*(i)}]$ (over applicable samples only) |
| **Majority baseline** | Always predict the most frequent class in training set |
| **Lift over baseline** | Accuracy − Majority baseline |

**Expected performance tiers** (from paper section 6.3):

| Tier | Features | Expected behaviour |
|---|---|---|
| High accuracy | primitive type, vacana, puruṣa | Tight agreement constraints → large lift over baseline |
| Moderate accuracy | vibhakti, lakāra, prayoga | Constrained but more freedom → meaningful lift |
| Low accuracy | upasarga, liṅga | Semantically driven → near baseline |

### 5. Loss convergence
- Training loss must decrease monotonically (on average) over epochs.
- Validation loss must decrease and eventually plateau (not diverge — no overfitting).

### 6. Oracle comparison
- When the true primitive type is provided as input to the output heads (oracle setting from paper section 6.1), accuracy on type-specific features (lakāra, vibhakti, etc.) should improve. This validates that the output heads correctly condition on type.

### 7. Ablation: MLP-only model (no attention)
- Train the MLP + output heads with a mean-pooled or last-position-only input instead of attended input.
- The accuracy gap between "with attention" and "without attention" quantifies the attention module's contribution.
- Expectation: agreement features (vacana, puruṣa) should show the largest drop without attention.

### 8. Search space reduction (practical metric)
- Given a predicted grammatical form, count how many primitives in the corpus are compatible with that form.
- Compare against total unique primitives of that type.
- Target: ~10× reduction in candidate set (paper section 7.1).
