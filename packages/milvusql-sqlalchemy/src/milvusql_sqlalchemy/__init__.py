"""milvusql-sqlalchemy -- a SQLAlchemy 2.0 dialect for Milvus."""

from __future__ import annotations

import importlib.metadata as _metadata

from milvusql_sqlalchemy.dialect import MilvusDialect
from milvusql_sqlalchemy.hybrid import hybrid_search, weighted
from milvusql_sqlalchemy.types import SPARSEVEC, VECTOR

# Single-sourced from the installed package's metadata (pyproject's
# `version`, which the release pipeline bumps) -- a hardcoded string
# here silently drifted from the released version.
try:
    __version__ = _metadata.version("milvusql-sqlalchemy")
except _metadata.PackageNotFoundError:  # pragma: no cover -- source tree only
    __version__ = "0.0.0"

__all__ = [
    "SPARSEVEC",
    "VECTOR",
    "MilvusDialect",
    "__version__",
    "hybrid_search",
    "weighted",
]
