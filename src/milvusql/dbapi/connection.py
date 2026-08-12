"""PEP 249 ``Connection`` wrapping one ``MilvusClient`` (D9: one
gRPC channel per pooled DBAPI connection, the shape both SQLAlchemy's
``QueuePool`` and Django's per-thread connections expect)."""

from __future__ import annotations

import typing as t

from pymilvus import MilvusClient

from milvusql.dbapi import errors
from milvusql.dbapi.cursor import Cursor


class Connection:
    """Owns the ``MilvusClient`` and the connection-level consistency
    level fallback (D6: a query's own ``CONSISTENCY LEVEL`` clause
    always wins; this is only used when a statement doesn't set one)."""

    def __init__(
        self,
        *,
        consistency_level: str | None = None,
        **client_kwargs: t.Any,  # noqa: ANN401 -- passthrough to MilvusClient
    ) -> None:
        self._client = MilvusClient(**client_kwargs)
        self.consistency_level = consistency_level
        self.closed = False

    def cursor(self) -> Cursor:
        if self.closed:
            msg = "connection is closed"
            raise errors.InterfaceError(msg)
        return Cursor(self)

    def close(self) -> None:
        if not self.closed:
            self._client.close()
            self.closed = True

    def commit(self) -> None:
        """No-op (D7): every mutation is already applied the moment
        ``pymilvus`` returns -- there is nothing left to commit."""

    def rollback(self) -> None:
        """Fails loudly (D7) rather than silently discarding the
        request: Milvus has no multi-statement rollback, and a no-op
        here would read as "rolled back" to code that trusts
        ``rollback()`` the way it would against a transactional
        database."""
        msg = (
            "Milvus has no transaction rollback -- each statement is "
            "applied as soon as it returns. Catch and compensate "
            "instead of relying on rollback()."
        )
        raise errors.NotSupportedError(msg)

    def __enter__(self) -> Connection:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def connect(
    uri: str = "http://localhost:19530",
    token: str = "",  # nosec B107 -- optional auth token, not a secret default
    db_name: str = "",
    consistency_level: str | None = None,
    **kwargs: t.Any,  # noqa: ANN401 -- passthrough to MilvusClient
) -> Connection:
    """PEP 249 module-level entry point. Every keyword ``MilvusClient``
    accepts (``user``/``password``/``timeout``/...) passes straight
    through -- this layer does not reinvent Milvus's own connection
    parameters."""
    return Connection(
        uri=uri,
        token=token,
        db_name=db_name,
        consistency_level=consistency_level,
        **kwargs,
    )


__all__ = ["Connection", "connect"]
