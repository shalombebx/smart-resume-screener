"""
app/core/config.py
──────────────────
Central configuration — reads from .env via pydantic-settings.
All tunable knobs live here; nothing is hard-coded elsewhere.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str = Field("text-embedding-3-large")
    chat_model: str = Field("gpt-4o")

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    debug: bool = Field(False)

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_dir: Path = Field(Path(".cache/embeddings"))
    cache_ttl_seconds: int = Field(86400)

    # ── Upload ────────────────────────────────────────────────────────────────
    max_file_size_mb: int = Field(10)
    max_resumes_per_batch: int = Field(20)

    # ── Scoring weights ───────────────────────────────────────────────────────
    weight_embedding: float = Field(0.50)
    weight_keyword: float = Field(0.20)
    weight_experience: float = Field(0.15)
    weight_education: float = Field(0.10)
    weight_bonus: float = Field(0.05)

    @field_validator("cache_dir", mode="before")
    @classmethod
    def ensure_cache_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — import this everywhere."""
    return Settings()
