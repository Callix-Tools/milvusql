"""milvusql-django -- a Django database backend for Milvus."""

from __future__ import annotations

import importlib.metadata as _metadata

from milvusql_django.expressions import hybrid_search, vector_search
from milvusql_django.fields import VectorField

# Single-sourced from the installed package's metadata (pyproject's
# `version`, which the release pipeline bumps) -- a hardcoded string
# here silently drifted from the released version.
try:
    __version__ = _metadata.version("milvusql-django")
except _metadata.PackageNotFoundError:  # pragma: no cover -- source tree only
    __version__ = "0.0.0"

__all__ = [
    "VectorField",
    "__version__",
    "hybrid_search",
    "vector_search",
]
