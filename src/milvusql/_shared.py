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
#: They're also, not coincidentally, exactly the calls Milvus refuses
#: to run against an unloaded collection (D2 revised: auto-``LOAD``
#: below reuses this same tuple instead of a second, easy-to-drift
#: list) -- ``insert``/``upsert``/``delete`` are plain writes that
#: don't touch query-node memory at all, confirmed directly against a
#: real server.
CONSISTENCY_AWARE_METHODS = ("search", "query", "hybrid_search")

#: What a successful call does to a connection's ``_loaded_collections``
#: set: ``True`` -> now loaded, ``False`` -> no longer loaded, absent
#: -> no change. One table shared by ``dbapi/cursor.py`` (sync) and
#: ``aio.py`` (async) instead of the same three method names hardcoded
#: twice.
LOAD_STATE_EFFECTS: dict[str, bool] = {
    "load_collection": True,
    "release_collection": False,
    "drop_collection": False,
}


def note_load_state(
    loaded: set[str], method: str, kwargs: dict[str, object]
) -> None:
    """Update ``loaded`` (a connection's ``_loaded_collections``) after
    ``method`` has actually run on the server -- called unconditionally
    from both ``_invoke``s after every RPC, not just the auto-``LOAD``
    path, so an explicit ``LOAD TABLE``/``RELEASE TABLE``/``DROP
    TABLE`` the caller issues by hand keeps the cache honest too."""
    effect = LOAD_STATE_EFFECTS.get(method)
    if effect is None:
        return
    name = kwargs.get("collection_name")
    if not isinstance(name, str):
        return
    if effect:
        loaded.add(name)
    else:
        loaded.discard(name)


__all__ = [
    "CONSISTENCY_AWARE_METHODS",
    "LOAD_STATE_EFFECTS",
    "note_load_state",
    "parse_cached",
]
