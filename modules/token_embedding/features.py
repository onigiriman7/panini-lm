"""
features.py — Grammatical feature vocabulary definitions for the Pāṇinian token embedding.

Defines the 9 grammatical features used in the paper (Section 3, Table 1):
    primitive_type, lakāra, puruṣa, vacana, prayoga, pada, vibhakti, liṅga, upasarga

Each nullable feature reserves index 0 for an explicit NULL class.
Total input dimensionality: d_input = 3 + 11 + 4 + 4 + 4 + 3 + 9 + 4 + 20 = 62
"""

from enum import IntEnum


# ---------------------------------------------------------------------------
# Feature 1: Primitive Type (3 values, no NULL — always present)
# ---------------------------------------------------------------------------

class PrimitiveType(IntEnum):
    """Type of Sanskrit primitive: verb root, nominal stem, or indeclinable."""
    DHAATU      = 0   # dhātu (verb root / tiṅanta)
    PRAATIPADIKA = 1  # prātipadika (nominal stem / subanta)
    AVYAYA      = 2   # avyaya (indeclinable)


# ---------------------------------------------------------------------------
# Feature 2: Lakāra — tense/mood (10 values + NULL)
# ---------------------------------------------------------------------------

class Lakara(IntEnum):
    """Lakāra (tense/mood) of a tiṅanta. NULL for non-dhātu primitives."""
    NULL       = 0
    LAT        = 1   # laṭ  — present indicative
    LIT        = 2   # liṭ  — perfect
    LUT        = 3   # luṭ  — periphrastic future
    LRT        = 4   # lṛṭ  — simple future
    LOT        = 5   # loṭ  — imperative
    LAN        = 6   # laṅ  — imperfect
    VIDHI_LIN  = 7   # vidhi-liṅ — optative
    ASHIR_LIN  = 8   # āśīr-liṅ — benedictive
    LUN        = 9   # luṅ  — aorist
    LRN        = 10  # lṛṅ  — conditional


# ---------------------------------------------------------------------------
# Feature 3: Puruṣa — person (3 values + NULL)
# ---------------------------------------------------------------------------

class Purusha(IntEnum):
    """Puruṣa (person) of a tiṅanta. NULL for non-dhātu primitives."""
    NULL      = 0
    PRATHAMA  = 1   # prathama-puruṣa (3rd person)
    MADHYAMA  = 2   # madhyama-puruṣa (2nd person)
    UTTAMA    = 3   # uttama-puruṣa  (1st person)


# ---------------------------------------------------------------------------
# Feature 4: Vacana — number (3 values + NULL)
# ---------------------------------------------------------------------------

class Vacana(IntEnum):
    """Vacana (number). Applies to dhātu and prātipadika. NULL for avyaya."""
    NULL        = 0
    EKAVACANA   = 1   # singular
    DVIVACANA   = 2   # dual
    BAHUVACANA  = 3   # plural


# ---------------------------------------------------------------------------
# Feature 5: Prayoga — voice (3 values + NULL)
# ---------------------------------------------------------------------------

class Prayoga(IntEnum):
    """Prayoga (voice/usage) of a tiṅanta. NULL for non-dhātu primitives."""
    NULL    = 0
    KARTARI = 1   # kartari prayoga — active
    KARMANI = 2   # karmaṇi prayoga — passive
    BHAVE   = 3   # bhāve prayoga   — impersonal


# ---------------------------------------------------------------------------
# Feature 6: Pada — verb paradigm class (2 values + NULL)
# ---------------------------------------------------------------------------

class Pada(IntEnum):
    """Pada (verb paradigm class). NULL for non-dhātu primitives."""
    NULL          = 0
    PARASMAIPADA  = 1   # parasmaipada
    ATMANEPADA    = 2   # ātmanepada


# ---------------------------------------------------------------------------
# Feature 7: Vibhakti — case (8 values + NULL)
# ---------------------------------------------------------------------------

class Vibhakti(IntEnum):
    """Vibhakti (case) of a subanta. NULL for non-prātipadika primitives."""
    NULL       = 0
    PRATHAMAA  = 1   # prathamā    — nominative
    DVITIIYAA  = 2   # dvitīyā    — accusative
    TRTIIYAA   = 3   # tṛtīyā     — instrumental
    CATURTHII  = 4   # caturthī   — dative
    PANCAMII   = 5   # pañcamī    — ablative
    SASTHII    = 6   # ṣaṣṭhī     — genitive
    SAPTAMII   = 7   # saptamī    — locative
    SAMBODHANA = 8   # sambodhana  — vocative


# ---------------------------------------------------------------------------
# Feature 8: Liṅga — gender (3 values + NULL)
# ---------------------------------------------------------------------------

class Linga(IntEnum):
    """Liṅga (gender) of a subanta. NULL for non-prātipadika primitives."""
    NULL        = 0
    PULLINGA    = 1   # pulliṅga    — masculine
    STRIILINGA  = 2   # strīliṅga   — feminine
    NAPUMSAKA   = 3   # napuṃsaka   — neuter


# ---------------------------------------------------------------------------
# Feature 9: Upasarga — verbal prefix (19 values + NULL)
# ---------------------------------------------------------------------------

class Upasarga(IntEnum):
    """Upasarga (verbal prefix) of a dhātu. NULL when no prefix / non-dhātu."""
    NULL   = 0
    PRA    = 1    # pra
    PARAA  = 2    # parā
    APA    = 3    # apa
    SAM    = 4    # sam
    ANU    = 5    # anu
    AVA    = 6    # ava
    NIS    = 7    # nis / niḥ
    DUR    = 8    # dur / dus
    VI     = 9    # vi
    AA     = 10   # ā
    NI     = 11   # ni
    ADHI   = 12   # adhi
    API    = 13   # api
    ATI    = 14   # ati
    SU     = 15   # su
    UD     = 16   # ud / ut
    ABHI   = 17   # abhi
    PRATI  = 18   # prati
    PARI   = 19   # pari


# ===================================================================
# Feature metadata: ordered list, sizes, and applicability
# ===================================================================

# Canonical ordering of all 9 features (matches paper Table 1).
FEATURE_ORDER = [
    "primitive_type",
    "lakara",
    "purusha",
    "vacana",
    "prayoga",
    "pada",
    "vibhakti",
    "linga",
    "upasarga",
]

# Enum class for each feature (same order as FEATURE_ORDER).
FEATURE_ENUMS = [
    PrimitiveType,
    Lakara,
    Purusha,
    Vacana,
    Prayoga,
    Pada,
    Vibhakti,
    Linga,
    Upasarga,
]

# Number of one-hot slots per feature (= len(Enum)).
FEATURE_SIZES = [len(enum) for enum in FEATURE_ENUMS]
# [3, 11, 4, 4, 4, 3, 9, 4, 20]

# Total one-hot dimensionality.
D_INPUT = sum(FEATURE_SIZES)  # 62

# Starting index of each feature's one-hot block within the 62-dim vector.
FEATURE_OFFSETS = []
_offset = 0
for size in FEATURE_SIZES:
    FEATURE_OFFSETS.append(_offset)
    _offset += size

# Which features are applicable to each primitive type.
# Features not in this set receive the NULL index (0).
APPLICABLE_FEATURES = {
    PrimitiveType.DHAATU: {
        "primitive_type", "lakara", "purusha", "vacana",
        "prayoga", "pada", "upasarga",
    },
    PrimitiveType.PRAATIPADIKA: {
        "primitive_type", "vacana", "vibhakti", "linga",
    },
    PrimitiveType.AVYAYA: {
        "primitive_type",
    },
}

# Mapping from vidyut upasarga SLP1 strings → our Upasarga enum.
UPASARGA_SLP1_MAP = {
    "pra":   Upasarga.PRA,
    "parA":  Upasarga.PARAA,
    "apa":   Upasarga.APA,
    "sam":   Upasarga.SAM,
    "anu":   Upasarga.ANU,
    "ava":   Upasarga.AVA,
    "nis":   Upasarga.NIS,
    "nir":   Upasarga.NIS,
    "niH":   Upasarga.NIS,
    "dus":   Upasarga.DUR,
    "dur":   Upasarga.DUR,
    "vi":    Upasarga.VI,
    "A":     Upasarga.AA,
    "ni":    Upasarga.NI,
    "aDi":   Upasarga.ADHI,
    "adhi":  Upasarga.ADHI,
    "api":   Upasarga.API,
    "ati":   Upasarga.ATI,
    "su":    Upasarga.SU,
    "ud":    Upasarga.UD,
    "ut":    Upasarga.UD,
    "aBi":   Upasarga.ABHI,
    "abhi":  Upasarga.ABHI,
    "prati": Upasarga.PRATI,
    "pari":  Upasarga.PARI,
    "upa":   Upasarga.NULL,  # not in our 19; map to NULL
}
