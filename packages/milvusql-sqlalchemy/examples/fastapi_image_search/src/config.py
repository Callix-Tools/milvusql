"""Configuration for the image-search example."""

from __future__ import annotations

import os

#: A real Milvus server, not Milvus Lite -- see the README for the
#: bundled `docker-compose.yaml` that brings one up.
DATABASE_URL = "milvusql+aio://root:Milvus@localhost:19530/default"

COLLECTION_NAME = "images"

#: "clip" (default): real image/text embeddings via `sentence-
#: transformers`' CLIP model -- what makes search semantically
#: meaningful. "deterministic": a hash-derived pseudo-embedding with
#: no ML dependency at all, for smoke-testing this app's wiring only
#: -- see `embeddings.py`'s module docstring.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "clip")

#: `clip-ViT-B-32`'s embedding dimension -- shared between its image
#: and text encoders, which is what makes text-to-image search
#: (`GET /search/text`) possible in the first place.
VECTOR_DIM = 512
