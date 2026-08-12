"""milvusql-sqlalchemy -- a SQLAlchemy 2.0 dialect for Milvus."""

from __future__ import annotations

from milvusql_sqlalchemy.dialect import MilvusDialect
from milvusql_sqlalchemy.hybrid import hybrid_search, weighted
from milvusql_sqlalchemy.types import SPARSEVEC, VECTOR

__version__ = "0.1.0"

__all__ = [
    "SPARSEVEC",
    "VECTOR",
    "MilvusDialect",
    "__version__",
    "hybrid_search",
    "weighted",
]
