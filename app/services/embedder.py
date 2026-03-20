"""
app/services/embedder.py
─────────────────────────
Generates and caches OpenAI text embeddings.

Features:
- Uses text-embedding-3-large (configurable)
- Disk-based SHA-256 keyed cache (JSON) to avoid redundant API calls
- Batch support for multiple texts
- Cosine similarity utility
"""

from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("Vector length mismatch")
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingService:
    """
    Thin wrapper around the OpenAI Embeddings API with transparent disk caching.
    """

    def __init__(self) -> None:
        self._client: Optional[OpenAI] = None
        self._cache_dir = Path(settings.CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── OpenAI client (lazy init so tests can run without key) ────────────────

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not settings.OPENAI_API_KEY:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. "
                    "Add it to your .env file before running analysis."
                )
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_key(self, text: str, model: str) -> str:
        payload = f"{model}::{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_from_cache(self, key: str) -> Optional[List[float]]:
        p = self._cache_path(key)
        if settings.ENABLE_EMBEDDING_CACHE and p.exists():
            try:
                with p.open("r") as f:
                    return json.load(f)
            except Exception:
                logger.warning("Cache read failed for key %s — re-fetching", key[:12])
        return None

    def _save_to_cache(self, key: str, vector: List[float]) -> None:
        if not settings.ENABLE_EMBEDDING_CACHE:
            return
        try:
            with self._cache_path(key).open("w") as f:
                json.dump(vector, f)
        except Exception as exc:
            logger.warning("Cache write failed: %s", exc)

    # ── Core embedding ─────────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        """
        Return the embedding vector for a single text.
        Reads from cache if available; otherwise calls OpenAI and caches.
        """
        model = settings.OPENAI_EMBEDDING_MODEL
        key   = self._cache_key(text, model)

        cached = self._load_from_cache(key)
        if cached is not None:
            logger.debug("Embedding cache HIT (key=%s…)", key[:12])
            return cached

        logger.debug("Embedding cache MISS — calling OpenAI API")
        # Truncate to 8191 tokens approximate (chars / 4) to avoid API error
        truncated = text[:32000]

        response  = self.client.embeddings.create(
            input=truncated,
            model=model,
        )
        vector = response.data[0].embedding

        self._save_to_cache(key, vector)
        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts, leveraging cache per item."""
        return [self.embed(t) for t in texts]

    # ── Similarity ─────────────────────────────────────────────────────────────

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Return cosine similarity ∈ [-1, 1] between two texts.
        Normalised to [0, 100] for display.
        """
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)
        raw   = _cosine_similarity(vec_a, vec_b)
        # Clamp to [0, 1] then scale to %
        return round(max(0.0, raw) * 100, 2)


# ── Singleton ──────────────────────────────────────────────────────────────────
embedding_service = EmbeddingService()
