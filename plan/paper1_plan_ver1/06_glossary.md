# Glossary: Sanskrit Grammar Terms for Non-Specialists

This glossary explains the Sanskrit grammatical terms used throughout the project in plain language, with English parallels where possible.

---

## Word Types

### Dhātu (धातु) — Verb root
The core meaning-unit of a verb, before any suffixes are added. Like "go" in English before it becomes "goes", "went", "going", "gone." Sanskrit has ~2,000 of these. Example: गम् (gam) = "to go."

### Prātipadika (प्रातिपदिक) — Nominal stem
The core of a noun or adjective before its case ending is added. Like "dog" before it becomes "dog's" or "dogs." Example: राम (rāma) = "Rāma."

### Subanta (सुबन्त) — Inflected noun/adjective
A prātipadika with a case ending attached. The finished noun form as it appears in a sentence. Like "dogs" or "dog's" in English. Example: रामस्य (rāmasya) = "of Rāma."

### Tiṅanta (तिङन्त) — Conjugated verb
A dhātu with tense/person/number suffixes attached. The finished verb form. Like "runs" or "ran" in English. Example: गच्छति (gacchati) = "he/she goes."

### Avyaya (अव्यय) — Indeclinable
A word that never changes form — no case endings, no conjugation. Like "and", "but", "not" in English. Example: च (ca) = "and."

### Kṛdanta (कृदन्त) — Verbal derivative
A word formed from a verb root but functioning as a noun or adjective. Like "running" (from "run") in "the running water." Example: गतः (gataḥ) = "gone" (from गम्).

### Taddhita (तद्धित) — Secondary derivative
A word derived from another noun or adjective by adding a secondary suffix. Like "kingly" from "king" or "national" from "nation."

### Samāsa (समास) — Compound word
Two or more stems fused into one word. Sanskrit uses these very heavily. Like "blackbird" or "toothbrush" in English, but much more complex.

---

## Verb Features

### Lakāra (लकार) — Tense/mood system
Which tense or mood a verb is in. Sanskrit has 10 lakāras:

| Lakāra | Name | Meaning | English parallel |
|---|---|---|---|
| laṭ | लट् | Present | "he goes" |
| liṭ | लिट् | Perfect | "he has gone" |
| luṭ | लुट् | Periphrastic future | "he will go (definitely)" |
| lṛṭ | लृट् | Simple future | "he will go" |
| loṭ | लोट् | Imperative | "let him go!" |
| laṅ | लङ् | Imperfect | "he was going" |
| vidhi-liṅ | विधिलिङ् | Optative | "he should go" |
| āśīr-liṅ | आशीर्लिङ् | Benedictive | "may he go" (blessing) |
| luṅ | लुङ् | Aorist | "he went" (general past) |
| lṛṅ | लृङ् | Conditional | "he would have gone" |

### Puruṣa (पुरुष) — Person
Who is performing the action.

| Puruṣa | Sanskrit name | English parallel | Example |
|---|---|---|---|
| prathama | प्रथमपुरुष | Third person | "he/she/it goes" |
| madhyama | मध्यमपुरुष | Second person | "you go" |
| uttama | उत्तमपुरुष | First person | "I go" |

**Note**: Sanskrit ordering is opposite to English — third person is "first" (prathama).

### Prayoga (प्रयोग) — Voice
The relationship between the action, the agent, and the grammatical subject.

| Prayoga | Meaning | English parallel |
|---|---|---|
| kartari | Agent is subject | "Rāma eats the fruit" (active) |
| karmaṇi | Object is subject | "The fruit is eaten by Rāma" (passive) |
| bhāve | Action itself is subject | "There is eating by Rāma" (impersonal, rare in English) |

### Pada (पद) — Verb paradigm class
Which set of endings a verb takes. This is an inherent property of the verb root.

| Pada | Meaning |
|---|---|
| parasmaipada | "word for another" — action benefits another |
| ātmanepada | "word for oneself" — action benefits the agent |

Some roots can take both (ubhayapada).

### Upasarga (उपसर्ग) — Verbal prefix
A prefix attached to a verb root that modifies its meaning, like "un-", "re-", "over-" in English.

| Upasarga | Meaning shift | English analogy |
|---|---|---|
| ā | towards | "come" vs. "go" |
| vi | apart, specially | "dis-", "un-" |
| sam | together | "con-", "com-" |
| prati | back, against | "re-", "counter-" |
| anu | after, along | "follow along" |
| pari | around | "circum-" |

---

## Noun Features

### Vibhakti (विभक्ति) — Case
What grammatical role the noun plays in the sentence. Sanskrit has 8 cases:

| Vibhakti | Sanskrit name | Grammatical role | English parallel |
|---|---|---|---|
| prathamā | प्रथमा | Subject (nominative) | "Rāma goes" |
| dvitīyā | द्वितीया | Direct object (accusative) | "sees Rāma" |
| tṛtīyā | तृतीया | Instrument (instrumental) | "by/with Rāma" |
| caturthī | चतुर्थी | Recipient (dative) | "for/to Rāma" |
| pañcamī | पञ्चमी | Source (ablative) | "from Rāma" |
| ṣaṣṭhī | षष्ठी | Possession (genitive) | "of Rāma / Rāma's" |
| saptamī | सप्तमी | Location (locative) | "in/on/at Rāma" |
| sambodhana | सम्बोधन | Address (vocative) | "O Rāma!" |

### Vacana (वचन) — Number
How many entities the word refers to.

| Vacana | Meaning | English parallel |
|---|---|---|
| ekavacana | Singular | "one dog" |
| dvivacana | Dual | "two dogs" (English doesn't have this) |
| bahuvacana | Plural | "three or more dogs" |

**Note**: Sanskrit has a special form for exactly two — something English lacks entirely.

### Liṅga (लिङ्ग) — Gender
Grammatical gender of a noun. This is an inherent property of the noun, not the entity it refers to (same as French or German).

| Liṅga | Meaning |
|---|---|
| pulliṅga | Masculine |
| strīliṅga | Feminine |
| napuṃsakaliṅga | Neuter |

---

## Agreement Rules (Why Grammar Predicts Grammar)

### Kartā-Kriyā Agreement (Subject-Verb)
The verb must match its subject in puruṣa and vacana. This is the **strongest** agreement rule and the one most likely to be learned by our model.

```
रामः गच्छति          Rāma(eka) goes(prathama, eka)        ✓
रामौ गच्छतः          Rāma-two(dvi) go(prathama, dvi)      ✓
रामाः गच्छन्ति        Rāmas(bahu) go(prathama, bahu)       ✓
*रामः गच्छन्ति       Rāma(eka) go(prathama, bahu)         ✗ UNGRAMMATICAL
```

### Viśeṣaṇa-Viśeṣya Agreement (Adjective-Noun)
An adjective must match its noun in vibhakti, vacana, and liṅga.

```
सुन्दरः रामः          beautiful(m,eka,pra) Rāma(m,eka,pra)    ✓
सुन्दरं रामम्         beautiful(m,eka,dvi) Rāma(m,eka,dvi)    ✓
*सुन्दरा रामः        beautiful(f,eka,pra) Rāma(m,eka,pra)    ✗ GENDER MISMATCH
```

---

## Pāṇini's Aṣṭādhyāyī (अष्टाध्यायी)

A grammar treatise composed by Pāṇini around 400 BCE. In approximately 4,000 short rules (sūtras), it describes the entire generative system of Sanskrit — how to produce every valid word and sentence. It is often called the first formal grammar in human history, and its structure anticipates modern concepts in computer science (formal language theory, generative grammars).

The key insight for our project: Pāṇini's rules reference **formal categories** (dhātu, prātipadika, vibhakti, etc.), not word meanings. This structural property is what we test — whether these formal categories carry enough information to predict each other.
