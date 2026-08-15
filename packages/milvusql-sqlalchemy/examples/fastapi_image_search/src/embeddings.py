"""Pluggable image/text embedder.

Default backend is CLIP (`sentence-transformers`'s `clip-ViT-B-32`),
which embeds images and text into the *same* vector space -- that
shared space is what makes `GET /search/text` (finding images by a
text description) possible with the exact same vector column and
index as `GET /search/image`, not a separate text index.

A `deterministic` backend is also included, with no ML dependency at
all: a hash-derived pseudo-embedding, useful only for smoke-testing
this app's wiring end to end (insert, search, the HTTP surface)
without installing `sentence-transformers`/`torch` or downloading
model weights. It carries no real semantic meaning -- nearest
neighbors are nearest hashes, not nearest images. Select it with
`EMBEDDING_BACKEND=deterministic`.
"""

from __future__ import annotations

import hashlib
import math
import typing as t
from functools import lru_cache

from config import EMBEDDING_BACKEND, VECTOR_DIM


class Embedder(t.Protocol):
    def embed_image(self, data: bytes) -> list[float]: ...
    def embed_text(self, text: str) -> list[float]: ...


class ClipEmbedder:
    """Wraps `sentence-transformers`'s CLIP model. Loaded lazily, and
    only once: importing `sentence_transformers`/`torch` and pulling
    model weights is by far the most expensive part of handling the
    first request, so it happens on first use rather than at import
    time (keeps `EMBEDDING_BACKEND=deterministic` genuinely
    dependency-free, and keeps app startup fast either way)."""

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self._model_name = model_name
        self._model: t.Any = None

    def _get_model(self) -> t.Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_image(self, data: bytes) -> list[float]:
        from io import BytesIO  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        image = Image.open(BytesIO(data)).convert("RGB")
        vector = self._get_model().encode(image, normalize_embeddings=True)
        return vector.tolist()

    def embed_text(self, text: str) -> list[float]:
        vector = self._get_model().encode(text, normalize_embeddings=True)
        return vector.tolist()


class DeterministicEmbedder:
    """No ML dependency -- a hash-derived pseudo-embedding for
    smoke-testing this app's wiring only. See the module docstring."""

    def __init__(self, dim: int = VECTOR_DIM) -> None:
        self.dim = dim

    def _vector_from_bytes(self, data: bytes) -> list[float]:
        raw = bytearray()
        counter = 0
        while len(raw) < self.dim:
            raw += hashlib.sha256(data + counter.to_bytes(4, "big")).digest()
            counter += 1
        values = [b / 255.0 - 0.5 for b in raw[: self.dim]]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def embed_image(self, data: bytes) -> list[float]:
        return self._vector_from_bytes(data)

    def embed_text(self, text: str) -> list[float]:
        return self._vector_from_bytes(text.encode("utf-8"))


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if EMBEDDING_BACKEND == "deterministic":
        return DeterministicEmbedder()
    return ClipEmbedder()
