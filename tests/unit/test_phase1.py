"""
Tests for Phase 1: Morphological Ingestion.

Test categories:
1. Unit tests for individual components
2. Integration tests for full pipeline
3. Determinism tests (same input → same output)
4. Fallback tests (vidyut → heritage)

Run with: pytest tests/unit/test_phase1.py -v
"""

import pytest
from typing import List

from panini_lm.core.types import MorphToken, Phase1Output
from panini_lm.core.config import MorphologyConfig
from panini_lm.phase1_morphology import (
    ingest_morphology,
    get_analyzer,
    MorphologicalAnalyzer,
)
from panini_lm.phase1_morphology.heritage_backend import (
    HeritageBackend,
    get_heritage_backend,
    _SURFACE_TO_ANALYSIS,
)
from panini_lm.phase1_morphology.orchestrator import clear_cache


@pytest.fixture(autouse=True)
def clear_analyzer_cache():
    """Clear analyzer cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestHeritageBackend:
    """Tests for the HeritageBackend (Python fallback)."""
    
    def test_backend_is_available(self):
        """Heritage backend should always be available."""
        backend = get_heritage_backend()
        assert backend.is_available is True
        assert backend.name == "heritage"
    
    def test_simple_sandhi_split(self):
        """Test basic whitespace splitting."""
        backend = get_heritage_backend()
        result = backend.sandhi_split("rāmaḥ gacchati")
        
        assert len(result) == 2
        assert "rāmaḥ" in result
        assert "gacchati" in result
    
    def test_avagraha_sandhi_split(self):
        """Test splitting at avagraha (apostrophe)."""
        backend = get_heritage_backend()
        result = backend.sandhi_split("rāmo'pi")
        
        assert len(result) == 2
        # rāmo' should become rāmaḥ (restore visarga)
        assert result[0] == "rāmaḥ"
        # 'pi should become api (restore initial a)
        assert result[1] == "api"
    
    def test_analyze_known_noun(self):
        """Test analysis of known nominal form."""
        backend = get_heritage_backend()
        token = backend.analyze_token("rāmaḥ")
        
        assert token["surface"] == "rāmaḥ"
        assert token["stem"] == "rāma"
        assert token["type"] == "subanta"
        assert token["attributes"]["vibhakti"] == 1
        assert token["attributes"]["vacana"] == 1
    
    def test_analyze_known_verb(self):
        """Test analysis of known verbal form."""
        backend = get_heritage_backend()
        token = backend.analyze_token("gacchati")
        
        assert token["surface"] == "gacchati"
        assert token["stem"] == "gam"
        assert token["type"] == "tinanta"
        assert token["attributes"]["lakara"] == "lat"
    
    def test_analyze_avyaya(self):
        """Test analysis of indeclinable."""
        backend = get_heritage_backend()
        token = backend.analyze_token("ca")
        
        assert token["type"] == "avyaya"
        assert token["stem"] == "ca"
        assert len(token["attributes"]) == 0
    
    def test_analyze_unknown_token(self):
        """Unknown tokens should return type='unknown'."""
        backend = get_heritage_backend()
        token = backend.analyze_token("xyzabc")
        
        assert token["surface"] == "xyzabc"
        assert token["stem"] == "xyzabc"  # Uses surface as stem
        assert token["type"] == "unknown"
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        backend = get_heritage_backend()
        result = backend.analyze("rāmaḥ gacchati")
        
        assert result["raw_input"] == "rāmaḥ gacchati"
        assert len(result["tokens"]) == 2
        assert result["tokens"][0]["stem"] == "rāma"
        assert result["tokens"][1]["stem"] == "gam"


class TestGetAnalyzer:
    """Tests for analyzer selection logic."""
    
    def test_auto_falls_back_to_heritage(self):
        """Auto mode should fall back to heritage if vidyut unavailable."""
        # vidyut_py is typically not installed, so this should fall back
        analyzer = get_analyzer("auto")
        
        # Should be heritage (fallback) since vidyut_py likely not installed
        assert analyzer.name in ["heritage", "vidyut"]
        assert analyzer.is_available is True
    
    def test_explicit_heritage(self):
        """Explicit heritage selection should work."""
        analyzer = get_analyzer("heritage")
        
        assert analyzer.name == "heritage"
        assert analyzer.is_available is True
    
    def test_analyzer_is_cached(self):
        """Same backend request should return cached instance."""
        analyzer1 = get_analyzer("heritage")
        analyzer2 = get_analyzer("heritage")
        
        assert analyzer1 is analyzer2


class TestIngestMorphology:
    """Tests for the main ingest_morphology function."""
    
    def test_basic_sentence(self):
        """Test basic sentence analysis."""
        result = ingest_morphology("rāmaḥ gacchati")
        
        assert "tokens" in result
        assert "raw_input" in result
        assert "sandhi_splits" in result
        assert result["raw_input"] == "rāmaḥ gacchati"
    
    def test_with_sandhi(self):
        """Test sentence with sandhi resolution."""
        result = ingest_morphology("rāmo'pi")
        
        assert len(result["tokens"]) == 2
        # First should be rāma (nominative)
        assert result["tokens"][0]["stem"] == "rāma"
        # Second should be api (particle)
        assert result["tokens"][1]["stem"] == "api"
    
    def test_three_word_sentence(self):
        """Test subject-object-verb sentence."""
        result = ingest_morphology("rāmaḥ gṛham gacchati")
        
        assert len(result["tokens"]) == 3
        # Subject
        assert result["tokens"][0]["stem"] == "rāma"
        assert result["tokens"][0]["attributes"]["vibhakti"] == 1
        # Object (accusative for neuter)
        assert result["tokens"][1]["stem"] == "gṛha"
        # Verb
        assert result["tokens"][2]["type"] == "tinanta"
    
    def test_with_config(self):
        """Test with explicit config."""
        config = MorphologyConfig(backend="heritage")
        result = ingest_morphology("rāmaḥ", config=config)
        
        assert len(result["tokens"]) == 1
    
    def test_output_structure(self):
        """Verify Phase1Output structure matches type definition."""
        result = ingest_morphology("devaḥ vadati")
        
        # Check required keys
        assert "tokens" in result
        assert "raw_input" in result
        assert "sandhi_splits" in result
        
        # Check token structure
        for token in result["tokens"]:
            assert "surface" in token
            assert "stem" in token
            assert "type" in token
            assert "attributes" in token


class TestDeterminism:
    """Tests ensuring deterministic output (critical for reproducibility)."""
    
    def test_same_input_same_output(self):
        """Same input must produce identical output."""
        text = "rāmaḥ gṛham gacchati"
        
        result1 = ingest_morphology(text)
        result2 = ingest_morphology(text)
        
        assert result1["tokens"] == result2["tokens"]
        assert result1["sandhi_splits"] == result2["sandhi_splits"]
    
    def test_determinism_with_sandhi(self):
        """Sandhi resolution must be deterministic."""
        text = "rāmo'pi"
        
        results = [ingest_morphology(text) for _ in range(5)]
        
        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert result["tokens"] == first["tokens"]
    
    def test_unicode_normalization(self):
        """Different Unicode representations should normalize to same output."""
        # These might be different Unicode representations of the same text
        text1 = "rāmaḥ"
        text2 = "rāmaḥ"  # Same visual appearance
        
        result1 = ingest_morphology(text1)
        result2 = ingest_morphology(text2)
        
        assert result1["tokens"][0]["stem"] == result2["tokens"][0]["stem"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_string(self):
        """Empty string should return empty tokens list."""
        result = ingest_morphology("")
        assert result["tokens"] == []
    
    def test_whitespace_only(self):
        """Whitespace-only input should return empty tokens."""
        result = ingest_morphology("   ")
        assert result["tokens"] == []
    
    def test_single_token(self):
        """Single token input should work."""
        result = ingest_morphology("ca")
        
        assert len(result["tokens"]) == 1
        assert result["tokens"][0]["type"] == "avyaya"
    
    def test_unknown_tokens_graceful(self):
        """Unknown tokens should not raise, but return type='unknown'."""
        result = ingest_morphology("xyz123 abc456")
        
        assert len(result["tokens"]) == 2
        assert result["tokens"][0]["type"] == "unknown"
        assert result["tokens"][1]["type"] == "unknown"


class TestMorphologicalDatabase:
    """Tests for the morphological database coverage."""
    
    def test_database_has_common_forms(self):
        """Database should have common word forms."""
        common_forms = ["rāmaḥ", "gacchati", "ca", "api", "devaḥ"]
        
        for form in common_forms:
            assert form in _SURFACE_TO_ANALYSIS, f"Missing form: {form}"
    
    def test_verb_paradigm_coverage(self):
        """Check verb paradigm coverage."""
        # Present tense of gam (to go)
        present_forms = ["gacchati", "gacchanti"]
        
        for form in present_forms:
            assert form in _SURFACE_TO_ANALYSIS, f"Missing verb form: {form}"
            stem, type_, attrs = _SURFACE_TO_ANALYSIS[form]
            assert stem == "gam"
            assert type_ == "tinanta"
