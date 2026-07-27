"""
Unit tests for config.settings.

Per Roadmap Phase 0 testing strategy: "Unit tests: config loader raises on
missing vars." These tests are intentionally simple — the goal is to catch a
silent-None-propagation bug early, not to test pydantic itself.
"""

import pytest

from config.settings import Settings


def test_settings_loads_with_defaults():
    """Settings should construct without error even with no .env present,
    since Phase-1+ fields are optional at this stage of the project."""
    settings = Settings(_env_file=None)
    assert settings.api_env == "local"
    assert settings.azure_ai_search_index_name == "climateguard-rag-index"


def test_require_raises_clear_error_on_missing_field():
    settings = Settings(_env_file=None, fred_api_key=None)
    with pytest.raises(RuntimeError, match="Missing required setting 'fred_api_key'"):
        settings.require("fred_api_key")


def test_require_returns_value_when_present():
    settings = Settings(_env_file=None, fred_api_key="dummy-key-123")
    assert settings.require("fred_api_key") == "dummy-key-123"
