# Architecture Explained: How the Model Works

## The Intuition: A Grammatical Crystal Ball

Imagine you're a Sanskrit teacher watching a student write a sentence. The student has written three words so far, and you can see each word's grammatical properties — its case, number, person, etc. — but the words are hidden behind cards, so you can't see *which* words they are.

Can you guess the grammatical properties of the next word?

A good teacher can. If the student has written two nouns in the subject case (prathamā vibhakti) and both are plural, the teacher knows the verb that follows must also be plural. The teacher doesn't need to know *which* nouns — the grammar alone constrains the next step.

Our model is trying to learn to be that teacher.

---

## Step 1: Encoding — Turning Grammar into Numbers

### What the model sees

Each word in a sentence is represented **only** by its grammatical features, **never** by its identity. Think of it as a form with checkboxes:

```
Word position 1:
  ☑ Type: prātipadika (noun stem)
  ☑ Vibhakti: prathamā (subject case)
  ☑ Vacana: ekavacana (singular)
  ☑ Liṅga: pulliṅga (masculine)
  ☐ Lakāra: N/A (not a verb)
  ☐ Puruṣa: N/A (not a verb)
  ☐ Prayoga: N/A (not a verb)
  ☐ Pada: N/A (not a verb)
  ☐ Upasarga: N/A (not a verb)

Word position 2:
  ☑ Type: dhātu (verb root)
  ☐ Vibhakti: N/A (not a noun)
  ☑ Vacana: ekavacana (singular)
  ☐ Liṅga: N/A (not a noun)
  ☑ Lakāra: laṭ (present tense)
  ☑ Puruṣa: prathama (third person)
  ☑ Prayoga: kartari (active voice)
  ☑ Pada: parasmaipada
  ☐ Upasarga: none
```

### How it's encoded (one-hot vectors)

Each checkbox group becomes a list of 0s and 1s. For "primitive type" with 3 options (dhātu, prātipadika, avyaya), a prātipadika is encoded as:

```
dhātu:       [0]
prātipadika: [1]
avyaya:      [0]
```

All these lists are concatenated (glued end-to-end) into one long vector of 54 numbers — mostly 0s with a few 1s sprinkled where the checkboxes are ticked.

**Key point**: The word itself — whether it's "Rāma" or "Kṛṣṇa" or "dharma" — is nowhere in this representation. The model is grammatically informed but semantically blind.

---

## Step 2: Attention — Learning Which Positions Matter

### The problem

Not all prior words matter equally for predicting the next one. If the sentence has 10 words so far, the subject noun (word 2) might matter a lot for predicting the verb's person and number, while an adverb (word 7) might be irrelevant.

### How attention works (the dinner party analogy)

Imagine a dinner party where each guest (= word position) has a card showing their grammatical features. Each guest can look around the table and decide: "Whose card is most relevant to me?"

Attention is exactly this: each position computes a relevance score with every other position, then creates a weighted summary — paying more attention to the positions that seem most relevant.

Concretely, for each word position, the model computes three things:

- **Query** (Q): "What am I looking for?" — derived from this position's grammar
- **Key** (K): "What do I have to offer?" — derived from each other position's grammar
- **Value** (V): "What information do I carry?" — derived from each other position's grammar

The attention score between position $i$ and position $j$ is how well $i$'s Query matches $j$'s Key. Positions with high scores get more weight when combining Values.

```
Position 5 (verb, need to predict its features):
  "I'm looking for: a subject noun that determines my person/number"
  
Position 2 (noun, prathamā vibhakti, ekavacana):
  Attention score: HIGH — this is the subject, very relevant!
  
Position 3 (noun, tṛtīyā vibhakti):
  Attention score: LOW — this is the instrument, less relevant to verb agreement
```

### Multi-head attention: Looking at multiple things at once

The model uses multiple "attention heads" — each head can focus on a different kind of relationship:

- **Head 1** might learn to find subject-verb agreement patterns
- **Head 2** might learn to find adjective-noun agreement patterns
- **Head 3** might learn sentence-boundary patterns

This is like having multiple people each reading the same sentence but looking for different grammatical relationships.

### Why attention matters for this task

The Pāṇinian agreement rules operate over specific grammatical relationships (kartā-kriyā, viśeṣaṇa-viśeṣya) that span variable distances in a sentence. Attention lets the model learn to bridge these distances — to connect a subject noun with its verb even if many words intervene.

---

## Step 3: MLP — Combining the Evidence

### What it does

After attention has gathered relevant information from across the sentence, a two-layer feed-forward neural network (MLP) combines and transforms this information into something useful for prediction.

**Analogy**: If attention is like a research assistant who gathers relevant facts from many sources, the MLP is the analyst who synthesizes those facts into a conclusion.

```
Layer 1: Take the attended information → Apply learned weights → Apply ReLU activation
Layer 2: Take the result → Apply more learned weights → Produce a summary
```

ReLU is just a simple filter: keep positive values, set negative values to zero. It lets the network learn non-trivial patterns (combinations of features that matter).

---

## Step 4: Output Heads — Making Predictions

### One classifier per feature

Rather than predicting all grammatical features at once as a single mega-label, the model makes independent predictions for each feature dimension:

```
Summary from MLP
    ├── Head 1: "What primitive type?"    → softmax → [0.7 dhātu, 0.2 prātipadika, 0.1 avyaya]
    ├── Head 2: "What lakāra?"            → softmax → [0.8 laṭ, 0.05 liṭ, ...]
    ├── Head 3: "What puruṣa?"            → softmax → [0.9 prathama, 0.05 madhyama, 0.05 uttama]
    ├── Head 4: "What vacana?"            → softmax → [0.85 ekavacana, 0.1 dvivacana, 0.05 bahuvacana]
    ├── Head 5: "What prayoga?"           → softmax → [0.7 kartari, 0.2 karmaṇi, 0.1 bhāve]
    ├── Head 6: "What pada?"              → softmax → [0.6 parasmaipada, 0.4 ātmanepada]
    ├── Head 7: "What vibhakti?"          → softmax → [0.3 prathamā, 0.25 dvitīyā, ...]
    ├── Head 8: "What liṅga?"             → softmax → [0.4 pulliṅga, 0.3 strīliṅga, 0.3 napuṃsaka]
    └── Head 9: "What upasarga?"          → softmax → [0.6 NULL, 0.1 vi, ...]
```

**Softmax** converts raw scores into probabilities that sum to 1. The highest probability is the prediction.

### Why separate heads?

Because different features have different numbers of options and different prediction difficulties. Lumping them together would force the model to solve one enormous prediction problem. Separate heads let each feature be predicted independently, and let us measure accuracy per feature — which is exactly what we need to test our hypothesis.

### The NULL handling trick

When the model predicts "dhātu" (verb) as the next primitive type, only verb-relevant features matter (lakāra, puruṣa, prayoga, pada, upasarga). Noun features (vibhakti, liṅga) should be NULL. During training, we don't penalise the model for predictions on features that don't apply — they contribute zero to the loss.

---

## Step 5: Training — Learning from Examples

### What the model learns from

The model sees hundreds of sentences from the Bhagavad Gītā, each converted into a sequence of grammatical feature vectors. For each position in each sentence, it:

1. Looks at all prior positions' grammatical features
2. Predicts the grammatical features of the current position
3. Compares its prediction to the actual grammar
4. Adjusts its weights to reduce the error

### The loss function

The error (loss) is computed separately for each feature, then summed:

```
Total error = λ₁ × (type error) + λ₂ × (lakāra error) + λ₃ × (puruṣa error) + ...
```

The λ weights let us control how much each feature matters during training. Features that are rarer or harder might get higher weights so the model doesn't ignore them.

---

## Why This Architecture (And Not Something Else)

### Why not a simple lookup table?

Because agreement rules span variable distances. The subject might be 1 word before the verb, or 8 words before it. A lookup table can't handle variable-distance relationships.

### Why not an RNN/LSTM?

An RNN processes words one by one, left to right, gradually forgetting earlier words. Attention can reach back to any position with equal ease. In Sanskrit, where a sentence might front-load many nouns before the verb, the ability to reach back is crucial.

### Why not a full Transformer?

A full Transformer (like GPT) would work, but it would be massively over-parameterised for this task. We have ~700 sentences and only 54-dimensional inputs. A single attention layer with a small MLP is sufficient and won't overfit. If we used GPT-scale architecture, it would memorise the training data rather than learning genuine grammatical patterns.

### Why multi-head output instead of single classification?

A single classifier predicting the full grammatical vector would face a combinatorial explosion: 3 × 10 × 3 × 3 × 3 × 2 × 8 × 3 × 19 = 246,240 possible combinations. With only ~8,000 tokens, most combinations would never appear in training. Factoring into independent heads makes each individual prediction tractable.

---

## Model Size: Intentionally Small

This model is designed to be tiny:

- **Input**: 54 dimensions (one-hot encoded features)
- **Attention**: Single layer, 4 heads, ~64-dimensional internal representation
- **MLP**: Two layers, ~128 hidden units
- **Output**: 9 independent classifiers
- **Total parameters**: Roughly ~50,000 (vs. billions in GPT)

The smallness is a feature, not a limitation. If a tiny model can predict grammar from grammar, it means the signal is strong enough that brute-force pattern matching isn't needed — there are genuine, learnable regularities in the grammatical structure.
