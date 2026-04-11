"""
analyzer.py — Vidyut-based morphological analyzer for Sanskrit words.

Wraps vidyut.lipi (transliteration), vidyut.kosha (word lookup), and
vidyut.cheda (segmenter/fallback) to convert Devanagari Sanskrit words
into GrammaticalVector instances with the 9 Pāṇinian features.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import vidyut
from vidyut.lipi import transliterate, Scheme
from vidyut.kosha import Kosha
from vidyut.cheda import Chedaka
from vidyut.prakriya import (
    Lakara as VLakara,
    Purusha as VPurusha,
    Vacana as VVacana,
    Prayoga as VPrayoga,
    Linga as VLinga,
    Vibhakti as VVibhakti,
    DhatuPada as VDhatuPada,
)

from modules.token_embedding.features import (
    PrimitiveType, Lakara, Purusha, Vacana, Prayoga, Pada,
    Vibhakti, Linga, Upasarga, UPASARGA_SLP1_MAP,
)

logger = logging.getLogger(__name__)

# Default path for vidyut data (relative to project root).
DEFAULT_DATA_DIR = "vidyut-data"


# ===================================================================
# GrammaticalVector — the 9-feature representation of one primitive
# ===================================================================

@dataclass
class GrammaticalVector:
    """A single primitive's 9-dimensional grammatical feature vector.

    Each field is an integer index into its feature's Enum.
    NULL is always 0 for nullable features.
    """
    primitive_type: int   # PrimitiveType index (0–2)
    lakara:         int   # Lakara index        (0–10)
    purusha:        int   # Purusha index        (0–3)
    vacana:         int   # Vacana index         (0–3)
    prayoga:        int   # Prayoga index        (0–3)
    pada:           int   # Pada index           (0–2)
    vibhakti:       int   # Vibhakti index       (0–8)
    linga:          int   # Linga index          (0–3)
    upasarga:       int   # Upasarga index       (0–19)

    def as_tuple(self) -> tuple:
        """Return all 9 feature indices as a tuple (in FEATURE_ORDER)."""
        return (
            self.primitive_type, self.lakara, self.purusha,
            self.vacana, self.prayoga, self.pada,
            self.vibhakti, self.linga, self.upasarga,
        )

    def feature_labels(self) -> dict:
        """Return a human-readable dict of feature name → Enum member name."""
        from modules.token_embedding.features import FEATURE_ENUMS, FEATURE_ORDER
        labels = {}
        for name, enum_cls, idx in zip(FEATURE_ORDER, FEATURE_ENUMS, self.as_tuple()):
            labels[name] = enum_cls(idx).name
        return labels


# ===================================================================
# Vidyut enum → our enum mappings
# ===================================================================

_LAKARA_MAP = {
    VLakara.Lat:      Lakara.LAT,
    VLakara.Lit:      Lakara.LIT,
    VLakara.Lut:      Lakara.LUT,
    VLakara.Lrt:      Lakara.LRT,
    VLakara.Lot:      Lakara.LOT,
    VLakara.Lan:      Lakara.LAN,
    VLakara.VidhiLin: Lakara.VIDHI_LIN,
    VLakara.AshirLin: Lakara.ASHIR_LIN,
    VLakara.Lun:      Lakara.LUN,
    VLakara.Lrn:      Lakara.LRN,
}

_PURUSHA_MAP = {
    VPurusha.Prathama: Purusha.PRATHAMA,
    VPurusha.Madhyama: Purusha.MADHYAMA,
    VPurusha.Uttama:   Purusha.UTTAMA,
}

_VACANA_MAP = {
    VVacana.Eka:  Vacana.EKAVACANA,
    VVacana.Dvi:  Vacana.DVIVACANA,
    VVacana.Bahu: Vacana.BAHUVACANA,
}

_PRAYOGA_MAP = {
    VPrayoga.Kartari: Prayoga.KARTARI,
    VPrayoga.Karmani: Prayoga.KARMANI,
    VPrayoga.Bhave:   Prayoga.BHAVE,
}

_LINGA_MAP = {
    VLinga.Pum:       Linga.PULLINGA,
    VLinga.Stri:      Linga.STRIILINGA,
    VLinga.Napumsaka: Linga.NAPUMSAKA,
}

_VIBHAKTI_MAP = {
    VVibhakti.Prathama:   Vibhakti.PRATHAMAA,
    VVibhakti.Dvitiya:    Vibhakti.DVITIIYAA,
    VVibhakti.Trtiya:     Vibhakti.TRTIIYAA,
    VVibhakti.Caturthi:   Vibhakti.CATURTHII,
    VVibhakti.Panchami:   Vibhakti.PANCAMII,
    VVibhakti.Sasthi:     Vibhakti.SASTHII,
    VVibhakti.Saptami:    Vibhakti.SAPTAMII,
    VVibhakti.Sambodhana: Vibhakti.SAMBODHANA,
}

_PADA_MAP = {
    VDhatuPada.Parasmaipada: Pada.PARASMAIPADA,
    VDhatuPada.Atmanepada:   Pada.ATMANEPADA,
}


# ===================================================================
# MorphAnalyzer — the main analyzer class
# ===================================================================

class MorphAnalyzer:
    """Analyzes Sanskrit words using vidyut and produces GrammaticalVectors.

    Uses vidyut.kosha for direct word lookup, falling back to vidyut.cheda
    (segmenter) when kosha has no entry.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.info("Downloading vidyut data to %s ...", data_dir)
            vidyut.download_data(data_dir)

        self.kosha = Kosha(str(data_path / "kosha"))
        self.chedaka = Chedaka(str(data_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transliterate(self, text: str, source=Scheme.Devanagari, target=Scheme.Slp1) -> str:
        """Convert text between scripts (default: Devanagari → SLP1)."""
        return transliterate(text, source, target)

    def analyze(self, word_devanagari: str) -> Optional[GrammaticalVector]:
        """Analyze a Devanagari word and return its GrammaticalVector.

        Returns None if the word cannot be analyzed.
        """
        word_slp1 = self.transliterate(word_devanagari)
        return self.analyze_slp1(word_slp1)

    def analyze_slp1(self, word_slp1: str) -> Optional[GrammaticalVector]:
        """Analyze an SLP1-encoded word and return its GrammaticalVector.

        Tries kosha lookup first. Falls back to cheda segmenter.
        Returns None if the word cannot be analyzed.
        """
        # Strategy 1: Direct kosha lookup
        gv = self._try_kosha(word_slp1)
        if gv is not None:
            return gv

        # Strategy 2: Cheda segmenter fallback
        gv = self._try_cheda(word_slp1)
        if gv is not None:
            return gv

        logger.warning("Could not analyze word: %s", word_slp1)
        return None

    # ------------------------------------------------------------------
    # Private: kosha lookup
    # ------------------------------------------------------------------

    def _try_kosha(self, word_slp1: str) -> Optional[GrammaticalVector]:
        """Look up word in kosha. Returns the best GrammaticalVector or None."""
        try:
            entries = self.kosha[word_slp1]
        except KeyError:
            return None

        if not entries:
            return None

        # Prefer: avyaya > tinanta > subanta (heuristic for disambiguation).
        best = self._pick_best_entry(entries)
        return self._entry_to_vector(best)

    def _pick_best_entry(self, entries):
        """Pick the most likely PadaEntry from multiple kosha results.

        Priority: avyaya entries first, then tinanta (verb), then subanta (nominal).
        """
        avyaya_entries = [e for e in entries if e.is_avyaya]
        if avyaya_entries:
            return avyaya_entries[0]

        tinanta_entries = [e for e in entries if _is_tinanta(e)]
        if tinanta_entries:
            return tinanta_entries[0]

        # Default to first subanta
        return entries[0]

    # ------------------------------------------------------------------
    # Private: cheda fallback
    # ------------------------------------------------------------------

    def _try_cheda(self, word_slp1: str) -> Optional[GrammaticalVector]:
        """Use cheda segmenter as fallback. Takes the first token's analysis."""
        try:
            tokens = self.chedaka.run(word_slp1)
        except Exception as exc:
            logger.debug("Cheda failed on %s: %s", word_slp1, exc)
            return None

        if not tokens:
            return None

        # Use the first token that has data
        for token in tokens:
            if token.data is not None:
                return self._entry_to_vector(token.data)
        return None

    # ------------------------------------------------------------------
    # Private: PadaEntry → GrammaticalVector conversion
    # ------------------------------------------------------------------

    def _entry_to_vector(self, entry) -> GrammaticalVector:
        """Convert a vidyut PadaEntry (Tinanta or Subanta) to GrammaticalVector."""

        # Detect avyaya first
        if entry.is_avyaya:
            return GrammaticalVector(
                primitive_type=PrimitiveType.AVYAYA,
                lakara=Lakara.NULL, purusha=Purusha.NULL,
                vacana=Vacana.NULL, prayoga=Prayoga.NULL,
                pada=Pada.NULL, vibhakti=Vibhakti.NULL,
                linga=Linga.NULL, upasarga=Upasarga.NULL,
            )

        if _is_tinanta(entry):
            return self._tinanta_to_vector(entry)
        else:
            return self._subanta_to_vector(entry)

    def _tinanta_to_vector(self, entry) -> GrammaticalVector:
        """Convert a Tinanta (verb form) PadaEntry to GrammaticalVector."""
        lakara  = _LAKARA_MAP.get(entry.lakara, Lakara.NULL)
        purusha = _PURUSHA_MAP.get(entry.purusha, Purusha.NULL)
        vacana  = _VACANA_MAP.get(entry.vacana, Vacana.NULL)
        prayoga = _PRAYOGA_MAP.get(entry.prayoga, Prayoga.NULL)

        # Pada — from dhatu_entry.pada if available
        pada = Pada.NULL
        if entry.dhatu_entry and entry.dhatu_entry.pada is not None:
            pada = _PADA_MAP.get(entry.dhatu_entry.pada, Pada.NULL)

        # Upasarga — from dhatu.prefixes (take the first one)
        upasarga = Upasarga.NULL
        if entry.dhatu_entry and entry.dhatu_entry.dhatu.prefixes:
            first_prefix = entry.dhatu_entry.dhatu.prefixes[0]
            upasarga = UPASARGA_SLP1_MAP.get(first_prefix, Upasarga.NULL)
            if upasarga == Upasarga.NULL and first_prefix not in UPASARGA_SLP1_MAP:
                logger.debug("Unknown upasarga: %s", first_prefix)

        return GrammaticalVector(
            primitive_type=PrimitiveType.DHAATU,
            lakara=lakara, purusha=purusha,
            vacana=vacana, prayoga=prayoga,
            pada=pada,
            vibhakti=Vibhakti.NULL,
            linga=Linga.NULL,
            upasarga=upasarga,
        )

    def _subanta_to_vector(self, entry) -> GrammaticalVector:
        """Convert a Subanta (nominal form) PadaEntry to GrammaticalVector."""
        vibhakti = _VIBHAKTI_MAP.get(entry.vibhakti, Vibhakti.NULL)
        linga    = _LINGA_MAP.get(entry.linga, Linga.NULL)
        vacana   = _VACANA_MAP.get(entry.vacana, Vacana.NULL)

        return GrammaticalVector(
            primitive_type=PrimitiveType.PRAATIPADIKA,
            lakara=Lakara.NULL, purusha=Purusha.NULL,
            vacana=vacana, prayoga=Prayoga.NULL,
            pada=Pada.NULL, vibhakti=vibhakti,
            linga=linga, upasarga=Upasarga.NULL,
        )


# ===================================================================
# Utilities
# ===================================================================

def _is_tinanta(entry) -> bool:
    """Check if a PadaEntry is a Tinanta (verb form)."""
    return type(entry).__name__ == "PyPadaEntry_Tinanta"
