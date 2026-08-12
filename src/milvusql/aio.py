"""Asyncio-native client (D12): ``AsyncConnection``/``AsyncCursor`` over
``pymilvus.AsyncMilvusClient``.

Deliberately **not** PEP 249 -- ``Cursor.execute()`` cannot be a
coroutine and still be a PEP 249 cursor, so this is a separate public
surface, not a second personality bolted onto ``dbapi.Cursor``. It
shares everything sync/await-agnostic with the sync path: the same
``translate.ast_to_pymilvus.build_call`` dispatch table, the same
``dbapi.errors`` hierarchy and ``translate()`` function, the same parse
cache. The only thing that differs is the two lines that call the
client and ``await`` it.
"""

from __future__ import annotations

import typing as t

import grpc
from pymilvus import AsyncMilvusClient
from pymilvus.exceptions import MilvusException
from sqlglot.errors import SqlglotError

from milvusql._shared import CONSISTENCY_AWARE_METHODS, parse_cached
from milvusql.dbapi import errors
from milvusql.translate.ast_to_pymilvus import build_call


class AsyncCursor:
    """The async mirror of :class:`milvusql.dbapi.cursor.Cursor`. Same
    shape, same method names where they can be (``fetchone``/
    ``fetchmany``/``fetchall`` are coroutines here; everything else
    that touches no I/O -- ``description``, ``rowcount``,
    ``arraysize`` -- stays a plain attribute, exactly as it does on the
    sync cursor)."""

    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection
        self.description: list[tuple] | None = None
        self.rowcount = -1
        self.arraysize = 1
        self.closed = False
        self.lastrowid: t.Any = None
        self._rows: list[tuple[t.Any, ...]] = []
        self._index = 0

    def _check_open(self) -> None:
        if self.closed:
            msg = "cursor is closed"
            raise errors.InterfaceError(msg)

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> AsyncCursor:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def execute(
        self, operation: str, parameters: dict[str, t.Any] | None = None
    ) -> AsyncCursor:
        self._check_open()
        bound = dict(parameters or {})
        try:
            ast = parse_cached(operation)
            call = build_call(self.connection._client, ast, bound)
            default_level = self.connection.consistency_level
            if (
                default_level is not None
                and call.method in CONSISTENCY_AWARE_METHODS
                and "consistency_level" not in call.kwargs
            ):
                call.kwargs["consistency_level"] = default_level
            raw = await getattr(self.connection._client, call.method)(
                **call.kwargs
            )
            rows, description, rowcount, lastrowid = call.postprocess(raw)
        except (SqlglotError, MilvusException, grpc.RpcError) as exc:
            raise errors.translate(exc) from exc
        self.description = description
        self.rowcount = rowcount
        self.lastrowid = lastrowid
        self._rows = rows
        self._index = 0
        return self

    async def executemany(
        self, operation: str, seq_of_parameters: t.Iterable[dict[str, t.Any]]
    ) -> AsyncCursor:
        total = 0
        for parameters in seq_of_parameters:
            await self.execute(operation, parameters)
            if self.rowcount > 0:
                total += self.rowcount
        self.rowcount = total
        return self

    async def fetchone(self) -> tuple[t.Any, ...] | None:
        self._check_open()
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    async def fetchmany(
        self, size: int | None = None
    ) -> list[tuple[t.Any, ...]]:
        self._check_open()
        size = self.arraysize if size is None else size
        chunk = self._rows[self._index : self._index + size]
        self._index += len(chunk)
        return chunk

    async def fetchall(self) -> list[tuple[t.Any, ...]]:
        self._check_open()
        chunk = self._rows[self._index :]
        self._index = len(self._rows)
        return chunk

    async def setinputsizes(self, sizes: t.Sequence[t.Any]) -> None:
        """Not applicable -- Milvus has no prepared-statement input
        binding to size ahead of time. Async per SQLAlchemy's
        ``AsyncIODBAPICursor`` protocol (unlike ``setoutputsize``,
        which that protocol declares sync)."""

    def setoutputsize(self, size: int, column: int | None = None) -> None:
        """Not applicable, same reason as ``setinputsizes``."""

    def __aiter__(self) -> AsyncCursor:
        # NOT `async def`: `async for` calls `__aiter__()` synchronously
        # and expects the async iterator back directly, not a coroutine
        # -- an `async def __aiter__` would make `some.__aiter__()`
        # return a coroutine object with no `__anext__`, breaking
        # `async for row in cursor`. Confirmed directly.
        return self

    async def __anext__(self) -> tuple[t.Any, ...]:
        row = await self.fetchone()
        if row is None:
            raise StopAsyncIteration
        return row


class AsyncConnection:
    """The async mirror of
    :class:`milvusql.dbapi.connection.Connection`. Same consistency-
    level fallback (D6), same no-op commit / loud rollback (D7)."""

    def __init__(
        self,
        *,
        consistency_level: str | None = None,
        **client_kwargs: t.Any,  # noqa: ANN401 -- passthrough to AsyncMilvusClient
    ) -> None:
        self._client = AsyncMilvusClient(**client_kwargs)
        self.consistency_level = consistency_level
        self.closed = False

    def cursor(self) -> AsyncCursor:
        if self.closed:
            msg = "connection is closed"
            raise errors.InterfaceError(msg)
        return AsyncCursor(self)

    async def close(self) -> None:
        if not self.closed:
            await self._client.close()
            self.closed = True

    async def commit(self) -> None:
        """No-op (D7), same reasoning as the sync ``Connection``."""

    async def rollback(self) -> None:
        """Fails loudly (D7), same reasoning as the sync ``Connection``."""
        msg = (
            "Milvus has no transaction rollback -- each statement is "
            "applied as soon as it returns. Catch and compensate "
            "instead of relying on rollback()."
        )
        raise errors.NotSupportedError(msg)

    async def __aenter__(self) -> AsyncConnection:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


def connect(
    uri: str = "http://localhost:19530",
    token: str = "",  # nosec B107 -- optional auth token, not a secret default
    db_name: str = "",
    consistency_level: str | None = None,
    **kwargs: t.Any,  # noqa: ANN401 -- passthrough to AsyncMilvusClient
) -> AsyncConnection:
    """Async mirror of :func:`milvusql.dbapi.connect`. Not a coroutine
    itself -- ``AsyncMilvusClient()`` construction does no I/O, same as
    the sync ``MilvusClient()`` -- so it can be called directly rather
    than awaited, matching ``AsyncMilvusClient``'s own constructor."""
    return AsyncConnection(
        uri=uri,
        token=token,
        db_name=db_name,
        consistency_level=consistency_level,
        **kwargs,
    )


__all__ = ["AsyncConnection", "AsyncCursor", "connect"]
