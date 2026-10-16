"""
app/services/embedder.py
─────────────────────────
Generates and caches local text embeddings via Ollama (default: nomic-embed-text).

Features:
- Uses the embedding model named in OLLAMA_EMBEDDING_MODEL via the local Ollama API
- Disk-based SHA-256 keyed cache (JSON) with TTL (CACHE_TTL_SECONDS)
- Cosine similarity utility
"""

from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Character limit before sending text to the embedding model. Kept conservative
# because local embedding models have smaller context windows than OpenAI's;
# raise it if your Ollama setup handles longer inputs without errors.
_MAX_CHARS = 8000


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("Vector length mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingService:
    """Thin wrapper using the OpenAI SDK to talk to Ollama's embeddings endpoint."""

    def __init__(self) -> None:
        self._client: Optional[OpenAI] = None
        self._cache_dir = Path(settings.CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Local Ollama client (lazy init, no OpenAI key required) ───────────────

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=settings.OLLAMA_BASE_URL,
                api_key="ollama",          # required by the SDK, ignored by Ollama
            )
        return self._client

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_key(self, text: str, model: str) -> str:
        return hashlib.sha256(f"{model}::{text}".encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_from_cache(self, key: str) -> Optional[List[float]]:
        p = self._cache_path(key)
        if settings.ENABLE_EMBEDDING_CACHE and p.exists():
            try:
                if settings.CACHE_TTL_SECONDS > 0 and time.time() - p.stat().st_mtime > settings.CACHE_TTL_SECONDS:
                    return None                       # expired → re-fetch and overwrite
                with p.open("r") as f:
                    return json.load(f)
            except Exception:
                logger.warning("Cache read failed for key %s — re-fetching", key[:12])
        return None

    def _save_to_cache(self, key: str, vector: List[float]) -> None:
        if not settings.ENABLE_EMBEDDING_CACHE:
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with self._cache_path(key).open("w") as f:
                json.dump(vector, f)
        except Exception as exc:
            logger.warning("Cache write failed: %s", exc)

    # ── Core embedding ─────────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        """Return the embedding vector for one text (cache first, then Ollama)."""
        model = settings.OLLAMA_EMBEDDING_MODEL
        truncated = text[:_MAX_CHARS]
        key = self._cache_key(truncated, model)

        cached = self._load_from_cache(key)
        if cached is not None:
            logger.debug("Embedding cache HIT (key=%s…)", key[:12])
            return cached

        logger.debug("Embedding cache MISS — calling local Ollama embeddings API")
        response = self.client.embeddings.create(input=truncated, model=model)
        vector = response.data[0].embedding

        self._save_to_cache(key, vector)
        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts, leveraging cache per item."""
        return [self.embed(t) for t in texts]

    # ── Similarity ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_percent(raw: float) -> float:
        """Cosine ∈ [-1, 1] → clamp to [0, 1] → percent (0–100)."""
        return round(max(0.0, raw) * 100, 2)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity of two texts as a 0–100 score."""
        if not text_a.strip() or not text_b.strip():
            return 0.0
        return self._to_percent(_cosine_similarity(self.embed(text_a), self.embed(text_b)))

    def similarity_to_vector(self, vector: List[float], text: str) -> float:
        """Like ``similarity`` but reuses an already-computed vector (e.g. the JD's)."""
        if not text.strip():
            return 0.0
        return self._to_percent(_cosine_similarity(vector, self.embed(text)))


# ── Singleton ──────────────────────────────────────────────────────────────────
embedding_service = EmbeddingService()