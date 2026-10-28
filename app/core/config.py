"""
app/core/config.py
──────────────────
Environment-driven settings (Pydantic v2 / pydantic-settings).
"""
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # ── OpenAI (kept for compatibility; not used when running on Ollama) ──────
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = Field(
        "text-embedding-3-large",
        validation_alias=AliasChoices("OPENAI_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
    )
    OPENAI_CHAT_MODEL: str = Field(
        "gpt-4o",
        validation_alias=AliasChoices("OPENAI_CHAT_MODEL", "CHAT_MODEL"),
    )
    OPENAI_MAX_TOKENS: int = 800
    OPENAI_TEMPERATURE: float = 0.2

    # ── Ollama (local models, OpenAI-compatible API) ──────────────────────────
    # 127.0.0.1 rather than "localhost": Ollama listens on IPv4 only, and
    # "localhost" can try IPv6 (::1) first. In Docker, set this to
    # http://host.docker.internal:11434/v1 via the compose `environment:` block.
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434/v1"
    OLLAMA_CHAT_MODEL: str = "llama3.2:1b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    ENABLE_AI_EXPLANATIONS: bool = False

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: list[str] = ["http://localhost:8000", "http://localhost:5173"]  # JSON list in .env

    CACHE_DIR: Path = Path(".cache/embeddings")
    CACHE_TTL_SECONDS: int = 86400
    ENABLE_EMBEDDING_CACHE: bool = True

    UPLOAD_DIR: Path = Path("data/uploads")
    EXPORT_DIR: Path = Path("exports")
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".docx"]

    MAX_FILE_SIZE_MB: int = 10
    MAX_RESUMES_PER_BATCH: int = 20

    WEIGHT_EMBEDDING: float = 0.50
    WEIGHT_KEYWORD: float = 0.20
    WEIGHT_EXPERIENCE: float = 0.15
    WEIGHT_EDUCATION: float = 0.10
    WEIGHT_BONUS: float = 0.05

    @model_validator(mode="after")
    def _weights_sum_to_one(self):
        total = (
            self.WEIGHT_EMBEDDING + self.WEIGHT_KEYWORD + self.WEIGHT_EXPERIENCE
            + self.WEIGHT_EDUCATION + self.WEIGHT_BONUS
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0 (got {total})")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()