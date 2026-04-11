# Token Embedding Module

Converts Sanskrit words into 62-dimensional one-hot grammatical feature vectors based on the 9 Pāṇinian features described in the paper (Section 4.1).

## Architecture

```
modules/token_embedding/
├── features.py    — Feature vocabulary definitions (Enums, constants)
├── analyzer.py    — Vidyut-based morphological analyzer
├── embedding.py   — One-hot encoding & sequence assembly
├── demo.py        — Terminal-runnable demonstration
└── __init__.py    — Public API exports
```

## Features (9 dimensions, 62 one-hot slots)

| Feature | Cardinality | Slots | Applicable to |
|---|---|---|---|
| primitive_type | 3 | 0–2 | all |
| lakāra | 10 + 1 NULL | 3–13 | dhātu |
| puruṣa | 3 + 1 NULL | 14–17 | dhātu |
| vacana | 3 + 1 NULL | 18–21 | dhātu, prātipadika |
| prayoga | 3 + 1 NULL | 22–25 | dhātu |
| pada | 2 + 1 NULL | 26–28 | dhātu |
| vibhakti | 8 + 1 NULL | 29–37 | prātipadika |
| liṅga | 3 + 1 NULL | 38–41 | prātipadika |
| upasarga | 19 + 1 NULL | 42–61 | dhātu |

NULL = explicit index 0 per feature. Every slot has exactly one 1.

## Setup

```bash
pip install -r requirements.txt
# vidyut data is auto-downloaded on first run
```

## Usage

### As a library

```python
from modules.token_embedding import MorphAnalyzer, encode_onehot, D_INPUT

analyzer = MorphAnalyzer()

# Analyze a word
gv = analyzer.analyze("धर्मक्षेत्रे")
print(gv.feature_labels())
# {'primitive_type': 'PRAATIPADIKA', 'lakara': 'NULL', 'purusha': 'NULL',
#  'vacana': 'EKAVACANA', 'prayoga': 'NULL', 'pada': 'NULL',
#  'vibhakti': 'SAPTAMII', 'linga': 'NAPUMSAKA', 'upasarga': 'NULL'}

# Encode to one-hot
vec = encode_onehot(gv)  # shape: (62,)
```

### CLI demo

```bash
python -m modules.token_embedding.demo
```

## Demo Output (first line of Gītā)

```
Line 1: धृतराष्ट्रः उवाच — हे सञ्जय, धर्मक्षेत्रे कुरुक्षेत्रे ...

Word (देव)             SLP1               Type           Features
──────────────────────────────────────────────────────────────────
धृतराष्ट्रः            DftarAzwraH        PRAATIPADIKA   vacana=EKAVACANA, vibhakti=PRATHAMAA, linga=PULLINGA
उवाच                   uvAca              DHAATU         lakara=LIT, purusha=PRATHAMA, vacana=EKAVACANA, prayoga=KARTARI
हे                     he                 AVYAYA         (no inflection)
सञ्जय                  saYjaya            DHAATU         lakara=LOT, purusha=MADHYAMA, vacana=EKAVACANA, prayoga=KARTARI
धर्मक्षेत्रे           Darmakzetre        PRAATIPADIKA   vacana=EKAVACANA, vibhakti=SAPTAMII, linga=NAPUMSAKA
कुरुक्षेत्रे           kurukzetre         PRAATIPADIKA   vacana=EKAVACANA, vibhakti=SAPTAMII, linga=NAPUMSAKA

One-hot matrix (█ = 1, · = 0):
                     PRI|    LAK    |PUR |VAC |PRA |PAD|   VIB   |LIN |        UPA
  धृतराष्ट्रः        ·█·|█··········|█···|·█··|█···|█··|·█·······|·█··|█···················
  उवाच               █··|··█········|·█··|·█··|·█··|█··|█········|█···|█···················
  हे                 ··█|█··········|█···|█···|█···|█··|█········|█···|█···················
```

## Dependencies

- **vidyut** ≥ 0.4.0 — Sanskrit morphological analysis (kosha lookup + cheda segmenter fallback)
- **numpy** ≥ 1.24.0 — One-hot vector operations
