"""Configuration, read once from the environment."""

from __future__ import annotations

import os

#: Milvus Lite (a local file) by default -- zero infrastructure to try
#: the app. Point at a real server with, e.g.:
#: DATABASE_URL="milvusql+aio://root:Milvus@localhost:19530/default"
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "milvusql+aio:///./image_search.db"
)
#: Schema bootstrap (`app.db.bootstrap_schema`) runs through the sync
#: dialect, not `+aio` -- see that function's docstring for why. Same
#: connection target either way, just the driver name.
SYNC_DATABASE_URL = DATABASE_URL.replace("milvusql+aio", "milvusql", 1)

COLLECTION_NAME = "images"

#: "clip" (default): real image/text embeddings via `sentence-
#: transformers`' CLIP model -- what makes search semantically
#: meaningful. "deterministic": a hash-derived pseudo-embedding with
#: no ML dependency at all, for smoke-testing this app's wiring only
#: (insert, search, HTTP surface) -- see `app.embeddings`'s module
#: docstring.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "clip")

#: `clip-ViT-B-32`'s embedding dimension -- shared between its image
#: and text encoders, which is what makes text-to-image search
#: (`GET /search/text`) possible in the first place.
VECTOR_DIM = 512
