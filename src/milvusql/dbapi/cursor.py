"""PEP 249 ``Cursor``: parse MilvusQL text, dispatch through
``translate.ast_to_pymilvus``, hand rows back. The only sync-specific
file in the request path -- everything it calls into
(``build_call``, ``errors.translate``) is shared verbatim with
``aio.AsyncCursor`` (D1/D12)."""

from __future__ import annotations

import typing as t

import grpc
from pymilvus.exceptions import MilvusException
from sqlglot.errors import SqlglotError

from milvusql._shared import CONSISTENCY_AWARE_METHODS, parse_cached
from milvusql.dbapi import errors
from milvusql.translate.ast_to_pymilvus import Call, build_call

if t.TYPE_CHECKING:
    from milvusql.dbapi.connection import Connection


class Cursor:
    """Not thread-safe, same as every DBAPI cursor -- one per
    concurrent caller, sharing the connection's ``MilvusClient``."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self.description: list[tuple] | None = None
        self.rowcount = -1
        self.arraysize = 1
        self.closed = False
        #: PEP 249 convention (sqlite3, MySQLdb): the primary key
        #: ``INSERT`` just assigned, ``None`` otherwise. Milvus assigns
        #: these server-side only for ``auto_id`` collections.
        self.lastrowid: t.Any = None
        self._rows: list[tuple[t.Any, ...]] = []
        self._index = 0

    def _check_open(self) -> None:
        if self.closed:
            msg = "cursor is closed"
            raise errors.InterfaceError(msg)

    def close(self) -> None:
        self.closed = True

    def _invoke(self, call: Call) -> t.Any:  # noqa: ANN401 -- a raw pymilvus response is genuinely any shape
        default_level = self.connection.consistency_level
        if (
            default_level is not None
            and call.method in CONSISTENCY_AWARE_METHODS
            and "consistency_level" not in call.kwargs
        ):
            call.kwargs["consistency_level"] = default_level
        return getattr(self.connection._client, call.method)(**call.kwargs)

    def execute(
        self, operation: str, parameters: dict[str, t.Any] | None = None
    ) -> Cursor:
        self._check_open()
        bound = dict(parameters or {})
        try:
            ast = parse_cached(operation)
            call = build_call(self.connection._client, ast, bound)
            raw = self._invoke(call)
            # `Call.then` drives a statement that needs a second RPC
            # (UPDATE: read, then upsert -- see `Call`'s docstring).
            # Most statements never set it, so this loop runs zero
            # times for them.
            while call.then is not None:
                next_call = call.then(raw)
                if next_call is None:
                    break
                call = next_call
                raw = self._invoke(call)
            rows, description, rowcount, lastrowid = call.postprocess(raw)
        except (SqlglotError, MilvusException, grpc.RpcError) as exc:
            raise errors.translate(exc) from exc
        self.description = description
        self.rowcount = rowcount
        self.lastrowid = lastrowid
        self._rows = rows
        self._index = 0
        return self

    def executemany(
        self, operation: str, seq_of_parameters: t.Iterable[dict[str, t.Any]]
    ) -> Cursor:
        total = 0
        for parameters in seq_of_parameters:
            self.execute(operation, parameters)
            if self.rowcount > 0:
                total += self.rowcount
        self.rowcount = total
        return self

    def fetchone(self) -> tuple[t.Any, ...] | None:
        self._check_open()
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    def fetchmany(self, size: int | None = None) -> list[tuple[t.Any, ...]]:
        self._check_open()
        size = self.arraysize if size is None else size
        chunk = self._rows[self._index : self._index + size]
        self._index += len(chunk)
        return chunk

    def fetchall(self) -> list[tuple[t.Any, ...]]:
        self._check_open()
        chunk = self._rows[self._index :]
        self._index = len(self._rows)
        return chunk

    def setinputsizes(self, sizes: t.Sequence[t.Any]) -> None:
        """Not applicable -- Milvus has no prepared-statement input
        binding to size ahead of time."""

    def setoutputsize(self, size: int, column: int | None = None) -> None:
        """Not applicable, same reason as ``setinputsizes``. Named
        singular (``setoutputsize``, not ``setoutputsizes``) -- that's
        the actual PEP 249 spelling, confirmed against the spec text;
        SQLAlchemy's own DBAPI protocol checks agree."""

    def __iter__(self) -> Cursor:
        return self

    def __next__(self) -> tuple[t.Any, ...]:
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row


__all__ = ["Cursor"]
