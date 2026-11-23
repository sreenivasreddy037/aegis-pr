"""Runtime configuration loaded from environment variables / `.env`.

Every knob that changes behaviour (LLM mode, vector store, sandbox limits) lives here so
that the rest of the codebase never reads `os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_REPO = PROJECT_ROOT / "fixtures" / "sample_repo"
FIXTURE_DIFF = FIXTURE_REPO / "sample_pr.diff"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM -------------------------------------------------------------------------
    mock_llm: bool = Field(
        default=False,
        description="Force the deterministic offline responders (no network, no API key).",
    )
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    llm_timeout_sec: float = 60.0

    # --- Retrieval -------------------------------------------------------------------
    qdrant_mode: Literal["local", "remote"] = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "aegispr_policies"
    embedding_backend: Literal["auto", "hash", "fastembed"] = "auto"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    policy_top_k: int = 4

    # --- Sandbox ---------------------------------------------------------------------
    sandbox_timeout_sec: int = Field(default=10, ge=1, le=300)
    sandbox_max_output_bytes: int = 64_000

    # --- Review policy ---------------------------------------------------------------
    impact_max_hops: int = Field(default=2, ge=1, le=4)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    ablation: Literal["none", "no_impact_graph", "no_sandbox"] = Field(
        default="none",
        description="Benchmark ablations: review only diffed files / skip sandbox execution.",
    )

    # --- API -------------------------------------------------------------------------
    allowed_repo_root: Path = PROJECT_ROOT / "fixtures"
    max_diff_bytes: int = 500_000

    @property
    def offline(self) -> bool:
        """True when the pipeline must not call any remote LLM."""
        return (
            self.mock_llm
            or self.openai_api_key is None
            or not self.openai_api_key.get_secret_value()
        )

    @property
    def mode(self) -> Literal["mock", "live"]:
        return "mock" if self.offline else "live"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
