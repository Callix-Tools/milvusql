"""PEP 249 exception hierarchy, plus the translation boundary (D8).

Two upstream exception families would otherwise leak through
un-translated: ``sqlglot.errors.*`` (parsing MilvusQL text) and
``pymilvus.exceptions.MilvusException`` and its subclasses (talking to
the server). SQLAlchemy and Django both inspect the *type* of whatever a
DBAPI raises to decide things like "is this connection dead, should the
pool discard it" -- a foreign exception type defeats that. ``translate()``
is the single place that maps both families onto this hierarchy; both
the sync ``Cursor`` and the async ``aio`` client call it at every
boundary with ``pymilvus``/``sqlglot``, so the mapping lives once.
"""

from __future__ import annotations

import grpc
from pymilvus.exceptions import (
    CollectionNotExistException,
    ConnectError,
    ErrorCode,
    IndexNotExistException,
    MilvusException,
    MilvusUnavailableException,
    ParamError,
    SchemaNotReadyException,
)
from sqlglot.errors import (
    ParseError,
    SqlglotError,
    TokenError,
    UnsupportedError,
)


class Warning(Exception):  # noqa: A001,N818 -- PEP 249 mandates this name
    """Non-fatal condition worth surfacing to the caller."""


class Error(Exception):
    """Base of the PEP 249 hierarchy every other error inherits."""


class InterfaceError(Error):
    """Error in the DBAPI layer itself, not the database."""


class DatabaseError(Error):
    """Error reported by Milvus; the generic fallback below this."""


class DataError(DatabaseError):
    """Problem with the data itself (out-of-range, malformed, ...)."""


class OperationalError(DatabaseError):
    """Milvus is unreachable, or the operation failed at runtime."""


class IntegrityError(DatabaseError):
    """A constraint (primary key, schema) was violated."""


class InternalError(DatabaseError):
    """Milvus reported an internal error, not caused by the request."""


class ProgrammingError(DatabaseError):
    """The MilvusQL text or its target is wrong: bad syntax, unknown
    collection/index, a collection that needs loading first."""


class NotSupportedError(DatabaseError):
    """The request is well-formed but Milvus cannot do it (rejected
    ``ALTER``, an unsupported operator transpiled from a foreign
    dialect, ...)."""


#: Exception types from ``pymilvus`` that mean "not reachable / not
#: usable right now", not "the request was wrong".
_OPERATIONAL_TYPES = (MilvusUnavailableException, ConnectError)

#: Exception types from ``pymilvus`` that mean "the request names
#: something that doesn't exist or isn't ready", which a caller can fix
#: by changing the request -- same bucket as a bad MilvusQL statement.
_PROGRAMMING_TYPES = (
    CollectionNotExistException,
    IndexNotExistException,
    SchemaNotReadyException,
    ParamError,
)

#: Milvus reports "collection not loaded" as a plain ``MilvusException``
#: with no dedicated subclass. Matched on message text since the
#: server gives us nothing more specific to isinstance-check.
#:
#: D2 revised: ``Cursor``/``AsyncCursor`` now auto-``LOAD`` a
#: collection the first time a connection runs
#: ``search``/``query``/``hybrid_search`` against it (see
#: ``_shared.CONSISTENCY_AWARE_METHODS`` and
#: ``Connection._loaded_collections``), so this marker should be rare
#: in practice. It still fires for real: a bare ``MilvusClient`` call
#: that bypasses this DBAPI layer can release a collection out from
#: under a connection's cache, and auto-LOAD itself can legitimately
#: fail this way (no index yet, collection dropped concurrently) --
#: this mapping is what turns that failure into a ``ProgrammingError``
#: instead of a bare ``DatabaseError``. Explicit ``LOAD TABLE`` is
#: still available for callers who want to control replica count or
#: warm a collection up ahead of traffic.
_NOT_LOADED_MARKER = "not loaded"

#: Server-reported "doesn't exist" errors come back as a plain
#: ``MilvusException`` with one of these numeric codes, *not* as
#: ``CollectionNotExistException``/``IndexNotExistException`` --
#: confirmed directly (a real "collection does not exist" from Milvus
#: Lite carries ``code=100``/``ErrorCode.COLLECTION_NOT_FOUND`` on a
#: bare ``MilvusException``, so the isinstance checks above miss it).
_PROGRAMMING_CODES = {
    ErrorCode.COLLECTION_NOT_FOUND,
    ErrorCode.INDEX_NOT_FOUND,
}

#: Some RPCs (``add_collection_field`` on Milvus Lite, confirmed
#: directly) fail at the gRPC transport layer before Milvus's own
#: ``check_status`` ever runs, so they surface as a bare
#: ``grpc.RpcError`` instead of a ``MilvusException`` -- a real,
#: separate exception family this layer has to translate too.
_GRPC_OPERATIONAL_CODES = (
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
)


def _translate_sqlglot_error(exc: SqlglotError) -> Error:
    if isinstance(exc, (ParseError, TokenError)):
        return ProgrammingError(str(exc))
    if isinstance(exc, UnsupportedError):
        return NotSupportedError(str(exc))
    return ProgrammingError(str(exc))


def _translate_milvus_exception(exc: MilvusException) -> Error:
    if _NOT_LOADED_MARKER in exc.message.lower():
        return ProgrammingError(str(exc))
    if exc.code in _PROGRAMMING_CODES:
        return ProgrammingError(str(exc))
    return DatabaseError(str(exc))


def _translate_grpc_error(exc: grpc.RpcError) -> Error:
    """Every concrete gRPC error (sync ``_InactiveRpcError``, async
    ``AioRpcError``) is also a ``grpc.Call``, which is where ``.code()``
    actually lives -- confirmed via ``_InactiveRpcError.__mro__``. The
    bare ``grpc.RpcError`` base class doesn't declare it, hence the
    ``isinstance`` check rather than calling it unconditionally."""
    code = exc.code() if isinstance(exc, grpc.Call) else None
    if code is grpc.StatusCode.UNIMPLEMENTED:
        return NotSupportedError(str(exc))
    if code in _GRPC_OPERATIONAL_CODES:
        return OperationalError(str(exc))
    return DatabaseError(str(exc))


def translate(exc: Exception) -> Error:
    """Map an upstream ``sqlglot``/``pymilvus``/``grpc`` exception onto
    the PEP 249 hierarchy above. Never raises; returns the mapped
    exception for the caller to ``raise ... from exc``."""
    if isinstance(exc, SqlglotError):
        return _translate_sqlglot_error(exc)
    if isinstance(exc, _OPERATIONAL_TYPES):
        return OperationalError(str(exc))
    if isinstance(exc, _PROGRAMMING_TYPES):
        return ProgrammingError(str(exc))
    if isinstance(exc, MilvusException):
        return _translate_milvus_exception(exc)
    if isinstance(exc, grpc.RpcError):
        return _translate_grpc_error(exc)
    return DatabaseError(str(exc))


__all__ = [
    "DataError",
    "DatabaseError",
    "Error",
    "IntegrityError",
    "InterfaceError",
    "InternalError",
    "NotSupportedError",
    "OperationalError",
    "ProgrammingError",
    "Warning",
    "translate",
]
