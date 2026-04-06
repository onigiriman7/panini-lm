# Module 2: Attention

## Purpose

Learn which prior grammatical positions in the context window are relevant for predicting the next primitive's grammatical form. This module implements the attention layer described in section 4.2 of the paper. It captures long-range agreement constraints — for example, that a prathama-vibhakti nominal constrains the puruṣa and vacana of a later verb.

---

## Inputs

| Input | Shape | Source |
|---|---|---|
| $\mathbf{X}$ | $(B, W, d_{\text{input}})$ | Token embedding module output |

where:
- $B$ = batch size
- $W$ = context window size (e.g. 8)
- $d_{\text{input}}$ = one-hot encoded grammatical vector dimension (54 or 62, see token embedding plan)

---

## Transformation

### Multi-head self-attention

The input is projected into queries, keys, and values via learned weight matrices:

$$Q = \mathbf{X} W_Q, \quad K = \mathbf{X} W_K, \quad V = \mathbf{X} W_V$$

where $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{input}} \times d_{\text{model}}}$.

Attention scores are computed as scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

where $d_k = d_{\text{model}} / h$ and $h$ is the number of heads.

### Multi-head split

The model uses $h = 4$ attention heads (per plan ver1). Each head operates on a $d_k$-dimensional slice:

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

Heads are concatenated and projected:

$$\mathbf{H} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

where $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

### Causal masking

Since this is a next-step prediction task, position $t$ must only attend to positions $\leq t$. A causal mask is applied to the attention scores before softmax:

$$\text{mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

### Output extraction

For next-step prediction, we need the representation at the **last position** of the context window (position $W$). The attended representation at this position aggregates information from all prior positions via the learned attention weights.

$$\mathbf{h}_{\text{out}} = \mathbf{H}[:, W-1, :]$$

---

## Hyperparameters

| Parameter | Symbol | Value (from paper plan) |
|---|---|---|
| Number of heads | $h$ | 4 |
| Model dimension | $d_{\text{model}}$ | TBD (must be divisible by $h$) |
| Key/query dimension | $d_k$ | $d_{\text{model}} / h$ |

---

## Outputs

| Output | Shape | Description |
|---|---|---|
| $\mathbf{h}_{\text{out}}$ | $(B, d_{\text{model}})$ | Attended representation at the last context position, ready for MLP |

Alternatively, if the MLP operates on the full sequence (not just last position):

| Output | Shape | Description |
|---|---|---|
| $\mathbf{H}$ | $(B, W, d_{\text{model}})$ | Full attended sequence (MLP then selects last position) |

**Decision to be made:** whether position extraction happens here or in the MLP module.

---

## KPIs / Correctness Metrics

### 1. Shape correctness
- Output shape must be $(B, d_{\text{model}})$ or $(B, W, d_{\text{model}})$ depending on design decision.
- Intermediate attention weight matrix must be $(B, h, W, W)$.

### 2. Causal mask validity
- Attention weights at position $i$ must be zero for all positions $j > i$.
- Verify: `attention_weights[:, :, i, j] == 0` for all $j > i$.
- The upper triangle of each attention matrix (per head, per batch) must be exactly zero.

### 3. Attention weight normalization
- For each query position $i$, attention weights across valid key positions must sum to 1:
  $$\sum_{j \leq i} \alpha_{ij} = 1$$

### 4. Gradient flow
- Gradients must flow through the attention layer to all input positions.
- No dead heads: each head should produce non-uniform attention weights on at least some inputs (can measure entropy of attention distributions).

### 5. Agreement pattern detection (functional test)
- Construct a synthetic input where position $t-3$ is a prathama-vibhakti, ekavacana prātipadika. Place a dhātu at position $t$.
- After training, the attention weight from position $t$ to position $t-3$ should be notably higher than to irrelevant positions.
- This tests whether the model learns kartā-kriyā (subject-verb) agreement patterns.

### 6. Ablation baseline
- When attention is removed (ablation 1 from paper section 6.2), the downstream prediction accuracy should drop, especially for agreement-constrained features (vacana, puruṣa).
- The delta between "with attention" and "without attention" quantifies the module's contribution.

### 7. Context window sensitivity
- Accuracy should increase as context window $W$ grows from 1 to 8 (ablation 2 from paper).
- Saturation point indicates effective grammatical dependency length.
