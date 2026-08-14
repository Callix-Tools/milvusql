"""Configuration, read once from the environment."""

from __future__ import annotations

import os

#: Milvus Lite (a local file) by default -- zero infrastructure to try
#: the agent. Point at a real server with, e.g.:
#: DATABASE_URL="milvusql+aio://root:Milvus@localhost:19530/default"
DATABASE_URL = os.environ.get("DATABASE_URL", "milvusql+aio:///./catalog.db")
#: Schema bootstrap and seeding run through the sync dialect, not
#: `+aio` -- see `catalog.db.bootstrap_schema`'s docstring for why.
#: Same connection target either way, just the driver name.
SYNC_DATABASE_URL = DATABASE_URL.replace("milvusql+aio", "milvusql", 1)

COLLECTION_NAME = "products"

#: "minilm" (default): real text embeddings via `sentence-
#: transformers`' `all-MiniLM-L6-v2` -- what makes `search_products`
#: semantically meaningful. "deterministic": a hash-derived pseudo-
#: embedding with no ML dependency at all, for smoke-testing the
#: agent's tool wiring only -- see `catalog.embeddings`'s module
#: docstring.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "minilm")
#: `all-MiniLM-L6-v2`'s embedding dimension.
VECTOR_DIM = 384

#: Any model string `pydantic-ai` accepts (`openai:gpt-4o-mini`,
#: `anthropic:claude-sonnet-4-5`, ...) -- see
#: https://ai.pydantic.dev/models/ for the full list and the API key
#: environment variable each provider expects.
AGENT_MODEL = os.environ.get("AGENT_MODEL", "openai:gpt-4o-mini")
