# Module 1: Token Embedding

## Purpose

Convert each primitive's grammatical feature vector into a fixed-size numerical input representation suitable for the attention layer. This module implements section 4.1 of the paper ("Input representation").

---

## Inputs

**Per primitive** — a grammatical feature vector $g_i$ consisting of 9 categorical features:

| Feature | Cardinality | Applicable to |
|---|---|---|
| primitive type | 3 | all (dhātu, prātipadika, avyaya) |
| lakāra | 10 + 1 NULL | dhātu |
| puruṣa | 3 + 1 NULL | dhātu |
| vacana | 3 + 1 NULL | dhātu, prātipadika (subanta) |
| prayoga | 3 + 1 NULL | dhātu |
| pada | 2 + 1 NULL | dhātu |
| vibhakti | 8 + 1 NULL | prātipadika (subanta) |
| liṅga | 3 + 1 NULL | prātipadika (subanta) |
| upasarga | 19 + 1 NULL | dhātu |

**Per sequence** — a context window of up to $W$ consecutive grammatical feature vectors:

$$G = (g_{t-W+1}, \ldots, g_{t})$$

Each $g_i$ is a tuple of 9 categorical values (integers indexing into their respective vocabularies).

---

## Transformation

### Step 1: One-hot encoding per feature

Each categorical feature $f_j^{(i)}$ is converted to a one-hot vector of length equal to that feature's cardinality (including the NULL class):

$$\text{onehot}(f_j^{(i)}) \in \{0, 1\}^{|V_j|}$$

where $|V_j|$ is the vocabulary size of feature $j$.

### Step 2: Concatenation

The one-hot vectors for all 9 features are concatenated into a single vector:

$$\mathbf{x}_i = [\text{onehot}(f_1^{(i)}) \| \text{onehot}(f_2^{(i)}) \| \cdots \| \text{onehot}(f_9^{(i)})]$$

### Total input dimensionality

$$d_{\text{input}} = 3 + 11 + 4 + 4 + 4 + 3 + 9 + 4 + 20 = 62$$

> **Decision (resolved):** Each nullable feature includes an explicit NULL class at index 0. This means every feature slot in the one-hot vector has exactly one 1 (including NULL positions), giving $d_{\text{input}} = 62$. The paper's section 4.1 has been updated accordingly.

### Step 3: Sequence assembly

The context window of $W$ embedded vectors is stacked into a matrix:

$$\mathbf{X} \in \mathbb{R}^{W \times d_{\text{input}}}$$

This is the input to the attention layer.

---

## Outputs

| Output | Shape | Description |
|---|---|---|
| $\mathbf{X}$ | $(W, d_{\text{input}})$ | Matrix of one-hot encoded grammatical vectors for the context window |

For a single training sample, the output is a 2D tensor. For a batch of $B$ samples:

$$\mathbf{X}_{\text{batch}} \in \mathbb{R}^{B \times W \times d_{\text{input}}}$$

---

## KPIs / Correctness Metrics

### 1. Dimensional correctness
- Output shape must be exactly $(B, W, d_{\text{input}})$.
- Each row $\mathbf{x}_i$ must have exactly $d_{\text{input}}$ elements.

### 2. One-hot validity
- For each feature slot within each $\mathbf{x}_i$, exactly one element is 1 and the rest are 0 (if using explicit NULL index), OR the slot is all-zeros (if using all-zeros NULL encoding).
- No feature slot should have more than one 1.

### 3. Round-trip consistency
- Encode a known grammatical vector $g_i$ into $\mathbf{x}_i$, then decode $\mathbf{x}_i$ back to $g_i$. The result must match exactly.
- This must hold for all 3 primitive types (dhātu with all features filled, prātipadika with nominal features filled, avyaya with all features NULL except type).

### 4. NULL handling
- A dhātu token must have NULL for vibhakti and liṅga.
- A prātipadika (subanta) token must have NULL for lakāra, puruṣa, prayoga, pada, and upasarga.
- An avyaya token must have NULL for all features except primitive type.
- Verify that the encoding of these NULL values matches the chosen convention.

### 5. Coverage
- Every grammatical feature value that appears in the annotated Gītā corpus must map to a unique position in the one-hot vector.
- No unknown values should appear at inference time (closed vocabulary).

### 6. Batch consistency
- Processing $B$ samples individually must produce the same result as processing them as a batch.
