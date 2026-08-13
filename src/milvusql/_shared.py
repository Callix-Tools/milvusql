"""The handful of bits genuinely shared, verbatim, between the sync
DBAPI (``dbapi/cursor.py``) and the async client (``aio.py``) -- kept
in one neutral module so neither imports the other's private names
(D1/D12: one dispatch table, two thin call-site layers)."""

from __future__ import annotations

import functools

import sqlglot
from sqlglot import exp

#: Parsing is pure and param-independent, so the same SQL text always
#: parses to the same AST -- safe to reuse across calls with different
#: bind parameters, and cheap insurance against re-parsing an
#: ``executemany()`` operation string on every row (D1).
_PARSE_CACHE_SIZE = 256


@functools.lru_cache(maxsize=_PARSE_CACHE_SIZE)
def parse_cached(sql: str) -> exp.Expression:
    return sqlglot.parse_one(sql, read="milvus")


#: ``search``/``query``/``hybrid_search`` are the only calls that read
#: a request-level consistency level; DDL and mutations don't take one.
CONSISTENCY_AWARE_METHODS = ("search", "query", "hybrid_search")

__all__ = ["CONSISTENCY_AWARE_METHODS", "parse_cached"]
