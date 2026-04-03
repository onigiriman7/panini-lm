# Core Hypothesis: What Are We Trying to Prove?

## The Big Question (Plain Language)

Imagine you're listening to someone speak in English, but instead of hearing the actual words, you only see their **grammatical labels** — like "noun, singular, subject" → "verb, past tense, third person" → "noun, singular, object."

Could you guess what kind of grammatical label comes next — without knowing what the words actually are?

In English, the answer is "sort of." In Sanskrit, we believe the answer is **much more strongly yes**, because Sanskrit grammar is extraordinarily rule-bound. Every word wears its grammar on its sleeve through its endings (suffixes), and the rules for how words relate to each other grammatically are explicit and well-documented.

**Our hypothesis: You can predict what grammatical shape the next word will take, just from the grammatical shapes of the words before it — without knowing which words they are.**

---

## Why Sanskrit Is Special Here

### Grammar is visible in the word itself

In English, "the dog runs" and "the dogs run" show agreement, but the grammar is partially hidden. In Sanskrit, every noun and verb is inflected — the ending of each word directly encodes its grammatical role:

- **रामः** (rāmaḥ) = "Rāma" as subject (prathamā vibhakti, ekavacana)
- **रामम्** (rāmam) = "Rāma" as object (dvitīyā vibhakti, ekavacana)
- **रामेण** (rāmeṇa) = "by Rāma" (tṛtīyā vibhakti, ekavacana)

The same root word changes its ending to signal exactly how it functions in the sentence. This means every word carries a rich grammatical tag that we can extract.

### Agreement rules are strict

Sanskrit has mandatory agreement rules:

1. **Subject-verb agreement**: If the subject is singular third-person, the verb **must** be singular third-person. No exceptions.
2. **Adjective-noun agreement**: An adjective must match its noun in case (vibhakti), number (vacana), and gender (liṅga).
3. **These rules propagate**: Knowing one word's grammar constrains what others can be.

### Pāṇini wrote the rules down ~2,400 years ago

Pāṇini's Aṣṭādhyāyī is essentially a generative grammar — a finite set of rules that produces all valid Sanskrit sentences. The claim embedded in this system is that **grammar is a self-contained machine**: it doesn't need to know the meaning of words to determine whether a sentence is structurally valid.

---

## The Three Sub-Hypotheses

### H1: Primitive type is highly predictable

**What this means**: Given a sequence of words tagged as "noun, noun, verb, noun...", the **type** of the next word (noun vs. verb vs. indeclinable) is predictable.

**Why we expect this**: Sanskrit sentence structure follows patterns. A sentence typically needs a subject (noun) and a verb. After a series of nouns describing who/what/where, a verb usually follows. Indeclinables (words like "and", "but", "not") appear in characteristic positions.

**Analogy**: It's like predicting whether the next piece in a jigsaw puzzle is an edge piece, a corner piece, or a middle piece — the surrounding pieces constrain what fits.

**Success criterion**: Accuracy significantly above the majority-class baseline (always guessing "noun", the most common type).

### H2: Agreement features are predictable

**What this means**: Features like vacana (number: singular/dual/plural) and puruṣa (person: first/second/third) of a verb should be predictable from the nouns that came before it.

**Why we expect this**: If you've seen a singular noun in the subject position, the verb **must** be singular. This is a hard grammatical rule, not a tendency. The model should learn to exploit this.

**Analogy**: If someone tells you "The cat ___", you know the next word must be a singular verb form ("runs", "sleeps") — not a plural one ("run", "sleep"). In Sanskrit, this constraint is even stronger and more pervasive.

**Success criterion**: Vacana and puruṣa accuracy should be substantially above baseline, especially for verbs following noun subjects.

### H3: Some features are genuinely unconstrained by prior grammar

**What this means**: Features like upasarga (verbal prefix, like "un-" or "re-" in English) and liṅga (gender) should be **hard** to predict from grammar alone, because they depend on which specific word the speaker chose — which is a semantic decision.

**Why we expect this**: The gender of a noun is an inherent property of that noun, not determined by grammatical context. Similarly, which prefix a verb carries is a lexical choice. These are semantic, not syntactic.

**Success criterion**: These features should be close to the majority-class baseline. This is equally important — it validates that our model is learning genuine grammatical structure, not just memorising sequences.

---

## Why This Matters

### For understanding language

If grammar is predictable from grammar alone, it means syntax carries its own information — it's not just a mirror of meaning. This validates what Pāṇini claimed: grammar is a self-sufficient system.

### For building Sanskrit tools

If you can predict the grammatical form of the next word:

1. Instead of searching through all ~2,000 verb roots, you only search roots that can take the predicted grammatical form
2. This is like narrowing a multiple-choice exam from 2,000 options to maybe 50 — an enormous practical speedup
3. The grammar prediction becomes a "first pass" that dramatically reduces computational cost

### For AI and linguistics

This is the first time anyone has tested whether grammatical form is predictable from grammatical form alone — in any language. Sanskrit is the ideal testbed because its grammar is so explicitly encoded.

---

## What Would Disprove the Hypothesis?

The hypothesis is **disproved** if:

- Accuracy on all features is at or below the majority-class baseline
- The model performs no better than randomly guessing the most common label every time
- Features that should be constrained by agreement rules (vacana, puruṣa) show no improvement over baseline

The hypothesis is **partially confirmed** if:

- Some features (type, vacana, puruṣa) beat baseline significantly
- Other features (upasarga, liṅga) remain near baseline
- This would suggest grammar partially determines next-step form, which is the most likely and most interesting outcome

The hypothesis is **strongly confirmed** if:

- Agreement features reach very high accuracy (>80%)
- The pattern of what's predictable vs. what isn't aligns with linguistic theory about which features are syntactically vs. semantically determined
