# Data Pipeline: From Raw Text to Training Tensors

## Overview

This document describes how raw Sanskrit text becomes the grammatical feature sequences that our model trains on. Each step is explained so that someone unfamiliar with Sanskrit NLP can understand what happens and why.

---

## The Pipeline at a Glance

```
गीता raw text (Devanagari)
    │
    ▼
Step 1: Transliteration → SLP1 encoding
    │
    ▼
Step 2: Sandhi splitting → Individual word forms
    │
    ▼
Step 3: Morphological analysis → Grammatical tags per word
    │
    ▼
Step 4: Feature extraction → Structured feature vectors
    │
    ▼
Step 5: Sequence construction → Training samples
    │
    ▼
Step 6: Tensor encoding → Model-ready data
```

---

## Step 1: Transliteration

### What and why

Sanskrit is traditionally written in Devanagari (देवनागरी) script, but computational tools work better with a romanised representation. SLP1 (Sanskrit Library Phonetic 1) is a one-to-one romanisation that maps each Devanagari character to exactly one ASCII character.

```
Devanagari:  धृतराष्ट्रः उवाच
SLP1:        DftarAzwraH uvAca
```

### Why SLP1 and not IAST?

IAST (International Alphabet of Sanskrit Transliteration) uses diacritics (ā, ī, ṣ) which create encoding issues. SLP1 uses only ASCII characters, making it unambiguous for software.

### Tool: vidyut.lipi

Part of the Vidyut toolkit by Ambuda. Handles bidirectional conversion between all major Sanskrit encodings.

---

## Step 2: Sandhi Splitting

### What sandhi is (the key challenge)

In Sanskrit, when words appear next to each other in a sentence, their endings and beginnings merge according to phonological rules called **sandhi**. This means that in written Sanskrit, it's often impossible to tell where one word ends and the next begins.

```
Before sandhi:  rāmaḥ + gacchati → rāmo gacchati    (visarga + g = o + g)
Before sandhi:  ca + atra → cātra                     (a + a = ā)
Before sandhi:  mahā + ṛṣi → maharṣi                  (ā + ṛ = ar)
```

Splitting sandhi is like un-contracting "don't" → "do not" and "I'm" → "I am" — but far more pervasive and complex. Nearly every word boundary in Sanskrit involves a sandhi transformation.

### Why this matters for us

We need individual words (primitives) to tag them with grammatical features. If we can't split the words apart, we can't analyse them.

### Our data advantage

The Gītā text in our corpus (`data/gita.txt`) is already written in **prose form with explicit word boundaries**. This means sandhi splitting has already been done manually:

```
Original verse: dharmakṣetre kurukṣetre samavetā yuyutsavaḥ
Our prose:      धर्मक्षेत्रे कुरुक्षेत्रे समवेताः युयुत्सवः
```

The manual splitting is higher quality than any automated tool would produce. For extending to other texts, we would use `vidyut.cheda` for automated splitting.

---

## Step 3: Morphological Analysis

### What this means

Each split word needs to be tagged with its full grammatical description. This is the most critical step — it produces the grammatical feature vectors that our model trains on.

```
Input word:    गच्छति (gacchati)
Output tags:   type=tiṅanta, dhātu=गम्, lakāra=laṭ, puruṣa=prathama,
               vacana=ekavacana, prayoga=kartari, pada=parasmaipada
```

### How morphological analysis works

Sanskrit morphology is **generative**: every valid word form can be derived from a root + a set of affixes according to Pāṇini's rules. Morphological analysis reverses this process:

```
gacchati
  = gam + śap (verb class marker) + tip (person-number suffix)
  = gam root + laṭ lakāra + prathama puruṣa + ekavacana

rāmasya
  = rāma + ṅas (genitive suffix)
  = rāma stem + ṣaṣṭhī vibhakti + ekavacana + pulliṅga
```

### Tool: vidyut.kosha

Vidyut's morphological dictionary can look up most standard Sanskrit word forms and return their grammatical analysis. Limitations:
- Rare or irregular forms may not be in the dictionary
- Ambiguous forms (same surface form, multiple possible analyses) require disambiguation
- ~5% of tokens in our corpus may be unparsable

### Handling ambiguity

Some words have multiple valid analyses:
```
"rāmasya" → unambiguous: ṣaṣṭhī ekavacana
"devān"   → could be: dvitīyā bahuvacana OR ṣaṣṭhī bahuvacana (in older forms)
```

Strategy: Use context to disambiguate when possible. When not possible, take the most common analysis or drop the token.

---

## Step 4: Feature Extraction

### What we extract per token

From the morphological analysis, we extract a fixed set of features:

```python
features = {
    "type":      one of [subanta, tiṅanta, avyaya, kṛdanta, taddhita, samāsa, none]
    "vibhakti":  one of [none, prathamā, dvitīyā, tṛtīyā, caturthī, 
                         pañcamī, ṣaṣṭhī, saptamī, sambodhana]
    "vacana":    one of [none, ekavacana, dvivacana, bahuvacana]
    "puruṣa":    one of [none, prathama, madhyama, uttama]
    # Additional features from paper (not yet in training data):
    "lakāra":    one of [none, laṭ, liṭ, luṭ, lṛṭ, loṭ, laṅ, 
                         vidhi-liṅ, āśīr-liṅ, luṅ, lṛṅ]
    "prayoga":   one of [none, kartari, karmaṇi, bhāve]
    "pada":      one of [none, parasmaipada, ātmanepada]
    "liṅga":     one of [none, pulliṅga, strīliṅga, napuṃsaka]
    "upasarga":  one of [none, ā, vi, sam, ni, niḥ, ud, anu, abhi, 
                         prati, pari, ava, adhi, api, apa, su, dur, ut]
}
```

### Current state of our data

Our `gita_training.json` currently includes:
- ✅ type (7 values including kṛdanta, taddhita, samāsa beyond the paper's 3)
- ✅ vibhakti (9 values)
- ✅ vacana (4 values)
- ✅ puruṣa (4 values)
- ❌ lakāra (not yet extracted)
- ❌ prayoga (not yet extracted)
- ❌ pada (not yet extracted)
- ❌ liṅga (not yet extracted)
- ❌ upasarga (not yet extracted)

### Gap analysis: What needs to be added

The paper proposes 9 feature dimensions. Our current data has 4. We need to extract 5 more. Priority order:

1. **lakāra** — Critical for verb classification, determines tense/mood
2. **liṅga** — Important for adjective-noun agreement testing
3. **prayoga** — Important but might be heavily skewed (most verbs are kartari)
4. **pada** — Available from dhātu database, can cross-reference
5. **upasarga** — Can be extracted by prefix-stripping from verb forms

---

## Step 5: Sequence Construction

### How sentences become training samples

Each sentence is a sequence of grammatical feature vectors. The model's job is: given positions 1 through t, predict position t+1.

```
Sentence: "हे सञ्जय धर्मक्षेत्रे कुरुक्षेत्रे समवेताः युयुत्सवः मामकाः पाण्डवाः च किम् अकुर्वन्"

Position 1:  [BOS marker]
Position 2:  [type=avyaya, vib=none, vac=none, pur=none]          "हे"
Position 3:  [type=subanta, vib=sambodhana, vac=eka, pur=none]    "सञ्जय"
Position 4:  [type=subanta, vib=saptamī, vac=eka, pur=none]      "धर्मक्षेत्रे"
Position 5:  [type=subanta, vib=saptamī, vac=eka, pur=none]      "कुरुक्षेत्रे"
Position 6:  [type=subanta, vib=prathamā, vac=bahu, pur=none]     "समवेताः"
...
Position 12: [type=tiṅanta, vib=none, vac=bahu, pur=prathama]    "अकुर्वन्"
Position 13: [EOS marker]

Training examples from this sentence:
  Input: [pos1]              → Target: pos2 features
  Input: [pos1, pos2]        → Target: pos3 features
  Input: [pos1, pos2, pos3]  → Target: pos4 features
  ...and so on
```

### Context window

With a context window of 8, the model only sees the last 8 positions. For long sentences, early positions are dropped from context. This is a hyperparameter to ablate.

### BOS/EOS tokens

Special tokens marking sentence start ([BOS]) and end ([EOS]):
- [BOS] has all features set to a special "boundary" value
- [EOS] serves as a prediction target: can the model predict when a sentence ends?

---

## Step 6: Tensor Encoding

### From features to numbers

Each feature value maps to an integer via the vocabulary:

```python
type_vocab    = {"subanta": 0, "tiṅanta": 1, "avyaya": 2, ...}
vibhakti_vocab = {"none": 0, "prathamā": 1, "dvitīyā": 2, ...}
# etc.
```

### One-hot encoding

Each integer is expanded into a one-hot vector:

```
type = "subanta" = 0 → [1, 0, 0, 0, 0, 0, 0]   (7 positions for 7 type values)
vibhakti = "saptamī" = 7 → [0, 0, 0, 0, 0, 0, 0, 1, 0]   (9 positions)
```

All one-hot vectors concatenated = one input vector per token position.

### Current dimensionality

With current features: 7 + 9 + 4 + 4 = 24 dimensions per token.

With all paper features: 7 + 9 + 4 + 4 + 11 + 4 + 3 + 4 + 19 = 65 dimensions per token.

(Paper says 54 because it uses 3 type values; our data has 7.)

---

## Data Quality Checks

### Checks to run before training

1. **Distribution sanity**: Does each feature have a reasonable distribution? No feature should be 99% one value.
2. **Agreement validation**: In sentences where we can identify subject-verb pairs, do they actually agree in vacana/puruṣa? This validates annotation quality.
3. **Coverage**: What % of tokens have complete feature annotations? Tokens with missing features should be flagged.
4. **Sentence length distribution**: Are there outlier sentences that might distort training?

### Known issues in our data

- Chapter 7 has 442 sentences (63% of data) vs. 24-71 for other chapters. This is likely because it includes sub-verses or was processed differently.
- Some features (lakāra, liṅga, etc.) are not yet extracted.
- The vocabulary includes special types (kṛdanta, taddhita, samāsa) not mentioned in the paper — decide whether to merge these into the paper's 3 types or keep the richer taxonomy.

---

## Implementation Priority

```
Priority 1 (Minimum Viable Experiment):
  ├── Use existing 4 features (type, vibhakti, vacana, puruṣa)
  ├── Build model with these features
  ├── Train and evaluate
  └── This tests H1 (type predictability) and H2 (agreement features)

Priority 2 (Full Paper):
  ├── Extract remaining 5 features (lakāra, prayoga, pada, liṅga, upasarga)
  ├── Retrain with all 9 features
  └── Run all ablations

Priority 3 (Extensions):
  ├── Cross-validation across chapters
  ├── Search space reduction measurement
  └── Extend to additional texts beyond Gītā
```

The key insight: **we can test the core hypothesis right now** with the 4 features already extracted. The remaining features strengthen the paper but aren't required for a proof of concept.
