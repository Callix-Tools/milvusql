"""Configuration for the catalog-agent example."""

from __future__ import annotations

import os

#: A real Milvus server, not Milvus Lite -- see the README for the
#: bundled `docker-compose.yaml` that brings one up.
DATABASE_URL = "milvusql+aio://root:Milvus@localhost:19530/default"

COLLECTION_NAME = "products"

#: "minilm" (default): real text embeddings via `sentence-
#: transformers`' `all-MiniLM-L6-v2` -- what makes `search_products`
#: semantically meaningful. "deterministic": a hash-derived pseudo-
#: embedding with no ML dependency at all, for smoke-testing the
#: agent's tool wiring only -- see `embeddings.py`'s module
#: docstring.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "minilm")
#: `all-MiniLM-L6-v2`'s embedding dimension.
VECTOR_DIM = 384

#: Any model string `pydantic-ai` accepts (`openai:gpt-4o-mini`,
#: `anthropic:claude-sonnet-4-5`, ...) -- see
#: https://ai.pydantic.dev/models/ for the full list and the API key
#: environment variable each provider expects.
AGENT_MODEL = os.environ.get("AGENT_MODEL", "openai:gpt-4o-mini")
