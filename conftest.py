"""Shared pytest setup: force deterministic offline mode for every test run."""

import os

os.environ.setdefault("MOCK_LLM", "true")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import pytest

from app.config import reset_settings_cache


@pytest.fixture(autouse=True)
def _fresh_settings():
    reset_settings_cache()
    yield
    reset_settings_cache()
