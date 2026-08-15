"""Pluggable text embedder for the catalog.

Default backend is `sentence-transformers`' `all-MiniLM-L6-v2` -- what
makes `search_products` (`agent.py`) semantically meaningful.

A `deterministic` backend is also included, with no ML dependency at
all: a hash-derived pseudo-embedding, useful only for smoke-testing
the agent's tool wiring end to end (search, lookup) without installing
`sentence-transformers`/`torch` or downloading model weights. It
carries no real semantic meaning -- don't use it for anything but
exercising the code. Select it with `EMBEDDING_BACKEND=deterministic`.
"""

from __future__ import annotations

import hashlib
import math
import typing as t
from functools import lru_cache

from config import EMBEDDING_BACKEND, VECTOR_DIM


class Embedder(t.Protocol):
    def embed_text(self, text: str) -> list[float]: ...


class MiniLMEmbedder:
    """Wraps `sentence-transformers`' `all-MiniLM-L6-v2`. Loaded
    lazily, and only once: pulling model weights on first use keeps
    `EMBEDDING_BACKEND=deterministic` genuinely dependency-free."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: t.Any = None

    def _get_model(self) -> t.Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        vector = self._get_model().encode(text, normalize_embeddings=True)
        return vector.tolist()


class DeterministicEmbedder:
    """No ML dependency -- a hash-derived pseudo-embedding for
    smoke-testing this example's wiring only. See the module
    docstring."""

    def __init__(self, dim: int = VECTOR_DIM) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> list[float]:
        data = text.encode("utf-8")
        raw = bytearray()
        counter = 0
        while len(raw) < self.dim:
            raw += hashlib.sha256(data + counter.to_bytes(4, "big")).digest()
            counter += 1
        values = [b / 255.0 - 0.5 for b in raw[: self.dim]]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if EMBEDDING_BACKEND == "deterministic":
        return DeterministicEmbedder()
    return MiniLMEmbedder()
