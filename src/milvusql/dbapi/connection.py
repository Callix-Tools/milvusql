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
        #: Collections this connection has already loaded, so
        #: ``Cursor._invoke`` (D2 revised: auto-``LOAD``) only pays for
        #: a real ``load_collection`` RPC once per collection instead
        #: of once per ``search``/``query``/``hybrid_search`` call.
        #: Deliberately connection-scoped, not process- or
        #: server-scoped: Milvus's own load state lives on the server
        #: (or, for Milvus Lite, in the embedded process), so a second
        #: ``Connection`` -- another pooled DBAPI connection, another
        #: process -- can't see this cache and will (harmlessly) issue
        #: its own first ``load_collection``, which is why the RPC
        #: itself still has to be idempotent and cheap on a hit,
        #: confirmed directly (~1s cold, ~2ms once already loaded).
        self._loaded_collections: set[str] = set()

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


def token_from_credentials(
    user: str | None, password: str | None
) -> str | None:
    """Milvus's ``token`` auth parameter is commonly ``"user:password"``
    (confirmed against pymilvus's own docs: ``token="root:Milvus"`` is
    the default-user credential, and ``token="user_1:P@ssw0rd"`` for a
    custom one) -- it is genuinely a username and password joined by a
    colon, not an opaque bearer token. Adapters that split a connection
    URL into separate ``user``/``password`` fields (``milvusql-sqlalchemy``,
    ``milvusql-django``) call this to reassemble what Milvus actually
    wants, instead of forwarding ``password`` alone and silently
    dropping ``user``."""
    if user and password:
        return f"{user}:{password}"
    if password:
        return password  # a full "user:token" pasted as the URL password
    return None


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
