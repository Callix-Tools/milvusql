"""milvusql-django -- a Django database backend for Milvus."""

from __future__ import annotations

from milvusql_django.expressions import hybrid_search, vector_search
from milvusql_django.fields import VectorField

__version__ = "0.1.0"

__all__ = [
    "VectorField",
    "__version__",
    "hybrid_search",
    "vector_search",
]
