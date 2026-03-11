"""Test configuration and fixtures for Panini-LM."""

import pytest
import torch

from panini_lm.core.config import PaniniConfig, get_small_config
from panini_lm.core.types import MorphToken, Phase1Output


@pytest.fixture
def small_config() -> PaniniConfig:
    """Small config for fast testing."""
    return get_small_config()


@pytest.fixture
def device() -> torch.device:
    """Get available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def sample_tokens() -> list[MorphToken]:
    """Sample morphological tokens for testing."""
    return [
        {
            "surface": "rāmaḥ",
            "stem": "rāma",
            "type": "subanta",
            "attributes": {"vibhakti": 1, "vacana": 1, "linga": "m"}
        },
        {
            "surface": "gṛham",
            "stem": "gṛha",
            "type": "subanta",
            "attributes": {"vibhakti": 2, "vacana": 1, "linga": "n"}
        },
        {
            "surface": "gacchati",
            "stem": "gam",
            "type": "tinanta",
            "attributes": {"purusa": 1, "vacana": 1, "lakara": "lat"}
        },
    ]


@pytest.fixture
def sample_phase1_output(sample_tokens) -> Phase1Output:
    """Sample Phase 1 output for testing."""
    return {
        "raw_input": "rāmaḥ gṛhaṃ gacchati",
        "sandhi_splits": ["rāmaḥ", "gṛham", "gacchati"],
        "tokens": sample_tokens,
    }


@pytest.fixture
def sample_sentence_pairs() -> list[tuple[str, list[str]]]:
    """Sample Sanskrit sentences with expected sandhi splits."""
    return [
        ("rāmaḥ gacchati", ["rāmaḥ", "gacchati"]),
        ("devāśca narāśca", ["devāḥ", "ca", "narāḥ", "ca"]),
        ("rāmo'pi", ["rāmaḥ", "api"]),
    ]
