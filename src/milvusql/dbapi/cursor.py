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

from milvusql._shared import (
    CONSISTENCY_AWARE_METHODS,
    note_load_state,
    parse_cached,
)
from milvusql.dbapi import errors
from milvusql.translate.ast_to_pymilvus import (
    Call,
    build_batch_call,
    build_call,
)

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
        if call.method in CONSISTENCY_AWARE_METHODS:
            if (
                default_level is not None
                and "consistency_level" not in call.kwargs
            ):
                call.kwargs["consistency_level"] = default_level
            # D2 revised: auto-LOAD -- search/query/hybrid_search need
            # a loaded collection server-side; load it transparently on
            # first use per connection instead of making every caller
            # issue LOAD TABLE by hand first. `LOAD TABLE` itself
            # remains available for explicit control (replica count,
            # warming a collection up before traffic arrives).
            name = call.kwargs.get("collection_name")
            loaded = self.connection._loaded_collections
            if name is not None and name not in loaded:
                self.connection._client.load_collection(name)
                loaded.add(name)
        raw = getattr(self.connection._client, call.method)(**call.kwargs)
        note_load_state(
            self.connection._loaded_collections, call.method, call.kwargs
        )
        return raw

    def _run(self, call: Call) -> None:
        """Run ``call`` (looping through ``Call.then`` for a statement
        that needs a second RPC), translate any upstream exception, and
        set every cursor-state attribute PEP 249 promises after
        ``execute()``. Shared by :meth:`execute` and
        :meth:`executemany`'s batched-``INSERT`` fast path -- both end
        at "one ``Call``, translate errors, unpack ``postprocess()``",
        they only differ in how the ``Call`` gets built."""
        try:
            raw = self._invoke(call)
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

    def execute(
        self, operation: str, parameters: dict[str, t.Any] | None = None
    ) -> Cursor:
        self._check_open()
        bound = dict(parameters or {})
        try:
            ast = parse_cached(operation)
            call = build_call(self.connection._client, ast, bound)
        except (SqlglotError, MilvusException, grpc.RpcError) as exc:
            raise errors.translate(exc) from exc
        self._run(call)
        return self

    def executemany(
        self, operation: str, seq_of_parameters: t.Iterable[dict[str, t.Any]]
    ) -> Cursor:
        self._check_open()
        seq = list(seq_of_parameters)
        if not seq:
            # Nothing to do -- and nothing to parse, matching every
            # other DBAPI's treatment of an empty executemany() batch.
            self.rowcount = 0
            return self
        try:
            ast = parse_cached(operation)
            batch = build_batch_call(ast, seq)
        except (SqlglotError, MilvusException, grpc.RpcError) as exc:
            raise errors.translate(exc) from exc
        if batch is not None:
            # Milvus has a native batched primitive for this statement
            # shape (currently: INSERT's `insert(data=[...])` accepts
            # every row in one RPC) -- one round trip instead of one
            # per parameter set.
            self._run(batch)
            return self
        # No batched primitive for this statement shape (UPDATE,
        # DELETE, ...) -- fall back to one execute() per parameter set,
        # same as every DBAPI without native batch support does.
        total = 0
        for parameters in seq:
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
