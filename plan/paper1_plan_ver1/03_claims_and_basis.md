# Claims and Their Basis: Why We Believe What We Believe

This document examines each major claim in the paper and provides intuitive justification and evidence for why it should hold.

---

## Claim 1: "Grammar operates independently of meaning"

### What this actually means

When Pāṇini wrote his grammar rules, he didn't write rules like "if you're talking about a king, use this verb form." He wrote rules like "if the subject is singular third-person, the verb takes this affix." The rules reference grammatical categories, never word meanings.

This is like traffic laws: "vehicles turning left must yield to oncoming traffic" doesn't care whether the vehicle is a truck or a sedan. The rule operates on the structural category ("vehicle turning left"), not the specific instance.

### Basis for the claim

**Linguistic basis**: Pāṇini's Aṣṭādhyāyī contains ~4,000 sūtras. Of these, the vast majority reference only formal grammatical categories (dhātu, prātipadika, pratyaya, etc.) — not word meanings. The few sūtras that reference meaning (called "arthapara" sūtras) are exceptions, not the rule. This formal character of the system has been noted by every major Sanskrit grammarian from Kātyāyana to Patañjali.

**Structural basis**: Consider these two sentences:
- रामः गच्छति (rāmaḥ gacchati) — "Rāma goes"
- कृष्णः गच्छति (kṛṣṇaḥ gacchati) — "Kṛṣṇa goes"

The grammatical structure is identical despite completely different meanings. The grammar hasn't changed at all — same subject case, same number, same person agreement. Grammar doesn't care who is going.

**Computational test**: If we strip away word identity and keep only grammatical features, can a model still find patterns? If yes, grammar has structure independent of meaning. If no, grammar is just noise once you remove the words.

---

## Claim 2: "Agreement rules propagate structure through sentences"

### What this actually means

A grammatical feature choice at one position in a sentence constrains what features are possible at other positions. This constraint propagation is the mechanism by which grammatical structure extends beyond individual words.

### Basis for the claim

**Hard agreement rules (कारक-क्रिया सम्बन्ध)**:

Rule: The verb agrees with its kartā (agent/subject) in puruṣa and vacana.

| Subject grammar | → | Required verb grammar |
|---|---|---|
| prathamā, ekavacana | → | prathama puruṣa, ekavacana |
| prathamā, dvivacana | → | prathama puruṣa, dvivacana |
| prathamā, bahuvacana | → | prathama puruṣa, bahuvacana |

This is a **deterministic** constraint. Given the subject's vacana, the verb's vacana is fully determined. The model should learn this with very high accuracy.

**Soft structural patterns**:

Not all constraints are deterministic. Some are statistical:
- After a sequence of subanta (nouns), a tiṅanta (verb) becomes increasingly likely
- Certain vibhaktis (cases) tend to appear before others in typical sentence ordering
- Avyaya (indeclinables like "ca" = "and") tend to follow the words they connect

These patterns are regularities, not rules. The model should learn them with moderate accuracy.

**Why Sanskrit propagation is stronger than in most languages**:

In English, agreement is limited: "he runs" but "they run" — only subject-verb number agreement is strict. In Sanskrit, agreement operates across:
- Subject-verb (puruṣa + vacana)
- Adjective-noun (vibhakti + vacana + liṅga)
- Relative-correlative clauses (vibhakti + vacana + liṅga across clause boundaries)

More agreement = more constraint propagation = more predictability from grammar alone.

---

## Claim 3: "Grammatical form of the next primitive is predictable from prior forms"

### What this actually means

If you show someone a sequence of grammatical tags — e.g., [noun-subject-singular, noun-object-singular, ...] — they should be able to make a better-than-random guess about the next tag.

### Basis for the claim

**Argument from sentence structure**:

Sanskrit sentences have characteristic structures. The most common is SOV (Subject-Object-Verb), though word order is flexible. Even with flexible order, certain patterns dominate:

```
Typical pattern:
  [subanta-prathamā] [subanta-dvitīyā] ... [tiṅanta]
  (subject nouns)    (object nouns)        (main verb)
```

After seeing a subject noun and an object noun, the probability of a verb increases sharply. This is a type-level prediction (predicting "verb" vs. "noun") that should be learnable.

**Argument from agreement**:

Once type is predicted, agreement rules constrain the specific features:
- Predicted type = tiṅanta → vacana and puruṣa are constrained by prior subject
- Predicted type = subanta → if it's an adjective, vibhakti/vacana/liṅga match the noun it modifies

**Argument from information theory**:

If grammatical features were completely random (independent of prior context), each feature would carry maximum entropy. But grammar imposes constraints that reduce entropy. Our model is attempting to capture this entropy reduction. Any accuracy above the majority-class baseline represents real predictive information.

**What majority-class baseline means**: If 65% of all words are subanta (nouns), a dumb model that always guesses "subanta" would be 65% accurate on primitive type. Our model needs to beat this by exploiting grammatical context.

---

## Claim 4: "The Gītā is well-suited to this task"

### What this actually means

The Bhagavad Gītā is a good test corpus, not a random choice.

### Basis for the claim

**Grammatical diversity**: The Gītā contains:
- Narrative sections (Sañjaya describing the battlefield)
- Dialogue (Kṛṣṇa and Arjuna conversing)
- Philosophical exposition (abstract concepts)
- Imperative/command forms (Kṛṣṇa instructing Arjuna)
- Questions and answers

This means it uses a wide variety of lakāras (tenses/moods), puruṣas (first, second, third person), and sentence structures. A corpus with only one style would test whether grammar is predictable *in that style*, not whether grammar is inherently structured.

**From our actual data** (7 chapters processed):
- 697 sentences, 8,324 tokens
- Type distribution: 5,409 subanta (65%), 1,783 tiṅanta (21%), 1,132 avyaya (14%)
- Vocabulary: 3,378 unique tokens
- Sentence length: 4-46 tokens (mean 13.9)

**Size concerns**: 697 sentences is small by modern ML standards. This is actually a feature of the experiment: if a tiny model can learn grammatical patterns from a small corpus, the signal must be strong. If we needed millions of sentences, it would suggest we were pattern-matching rather than learning genuine rules.

**Potential weakness**: The Gītā is a single text with a characteristic style. Results may not generalise to all Sanskrit literature. We should acknowledge this and suggest testing on other texts as future work.

---

## Claim 5: "A two-stage architecture reduces search space by an order of magnitude"

### What this actually means

If you're trying to predict the next Sanskrit word, there are ~2,000 possible verb roots and hundreds of noun stems. But if you first predict the grammatical form (e.g., "the next word is a verb, third-person, singular, present tense, active voice"), the number of possible words shrinks dramatically.

### Basis for the claim (with numbers)

**Stage 1: Predict grammatical form** (this paper)

Suppose the model predicts with 80% confidence:
- Type: tiṅanta (verb)
- Lakāra: laṭ (present)
- Puruṣa: prathama (3rd person)
- Vacana: ekavacana (singular)
- Prayoga: kartari (active)
- Pada: parasmaipada

**Stage 2: Search compatible roots**

From our dhātu database of 2,259 roots:
- ~1,132 are parasmaipada-compatible
- A specific laṭ + prathama + ekavacana + kartari + parasmaipada combination is producible by a much smaller subset
- Conservatively, perhaps 200-500 roots can produce any given specific form

So instead of searching 2,259 possibilities, stage 2 searches ~300. That's roughly a 7-8× reduction.

**More aggressive case**: If vibhakti/liṅga/vacana are also predicted for nouns, the search space for nominal stems is similarly reduced. A specific [prathamā, ekavacana, pulliṅga] combination rules out many prātipadika possibilities.

**Why "order of magnitude" is reasonable**: Across all feature predictions combined, each correct prediction roughly halves the search space. 3-4 correct predictions → 8-16× reduction → roughly one order of magnitude.

---

## Claim 6: "This is the first framing of Sanskrit grammar as a next-step prediction task"

### Basis for the claim

**Existing Sanskrit NLP work does the opposite**:

| Existing approach | Direction | Input → Output |
|---|---|---|
| Morphological analysis | word → tags | "gacchati" → [laṭ, prathama, ekavacana, kartari, parasmaipada] |
| POS tagging | word → type | "gacchati" → verb |
| Word segmentation | string → words | "rāmogacchati" → "rāmaḥ gacchati" |
| Dependency parsing | words → tree | sentence → syntactic structure |

**Our approach**:

| Our approach | Direction | Input → Output |
|---|---|---|
| Grammar-to-grammar prediction | tags → tags | [prior grammatical forms] → [next grammatical form] |

The critical difference: we never see the words. Every existing system conditions on the word form itself. We condition only on prior grammatical context.

**Why nobody has done this before**:

1. It requires a rich, explicit morphological annotation system (which Sanskrit uniquely has via Pāṇini)
2. It requires the hypothesis that grammar is self-sufficient (which is not obvious in most languages)
3. Most NLP work focuses on the word-to-tag direction because the practical goal is to analyse existing text, not predict structure

---

## Claim 7: "This validates the Pāṇinian view computationally"

### What this actually means

Pāṇini claimed (implicitly, through the structure of his grammar) that syntactic rules are a self-contained formal system. We're testing this claim empirically: if a model can predict grammatical form from grammatical form, it confirms that grammatical structure has internal coherence independent of lexical content.

### Basis for the claim

**What constitutes computational validation**:

The argument structure is:
1. Pāṇini's grammar operates on formal categories, not meanings (established fact)
2. If formal categories alone contain predictive structure (our experiment), then...
3. The formal system carries information that is not reducible to semantic content
4. This is exactly what Pāṇini's architecture assumes

**What this is NOT claiming**:
- We are NOT claiming to prove Pāṇini "right" in some absolute sense
- We are NOT claiming grammar is *completely* independent of meaning
- We ARE claiming that grammar has *enough* internal structure to be partially self-predicting
- This is an empirical result with degrees, not a binary yes/no

**Strength of the claim depends on the accuracy we achieve**:
- If accuracy is at baseline → claim is not supported
- If accuracy is moderately above baseline → grammar has some internal structure
- If accuracy on agreement features is very high → strong support for Pāṇinian formal independence

---

## Summary: Evidence Needed

| Claim | What would support it | What would undermine it |
|---|---|---|
| Grammar independent of meaning | Any above-baseline accuracy | Accuracy at baseline on all features |
| Agreement propagates structure | High accuracy on vacana/puruṣa | These features at baseline |
| Next form is predictable | Accuracy > baseline on most features | All features at baseline |
| Gītā is suitable | Diverse grammar in data | Skewed distribution with low variety |
| Two-stage reduces search | Demonstrated reduction factor | Too many roots per grammatical form |
| First framing | Literature review confirms | Someone has done this before |
| Validates Pāṇini | Agreement features highly predictable | No learnable structure in grammar |
