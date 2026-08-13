"""``MilvusDialect`` -- the SQLAlchemy 2.0 entry point.

``do_rollback`` *is* overridden here, and deliberately disagrees with
``milvusql.dbapi.Connection.rollback()`` (D7 makes that raise
``NotSupportedError`` loudly, on purpose, for anyone calling it
directly against the DBAPI). Confirmed directly: SQLAlchemy's engine
calls ``dialect.do_rollback()`` as routine pool bookkeeping --
``first_connect``, connection-return-to-pool -- with no relation to a
user ever asking to roll something back, and it must always succeed.
Every other non-transactional dialect makes this a no-op at exactly
this layer (``elasticsearch-dbapi``'s ``BaseESDialect.do_rollback`` is
a bare ``pass``, read directly during design). D7's warning is real and
still enforced -- it just lives one layer down, where a caller's intent
is unambiguous.
"""

from __future__ import annotations

import contextlib
import inspect
import typing as t

from sqlalchemy import event, exc, pool
from sqlalchemy.engine import default
from sqlalchemy.sql import schema
from sqlalchemy.util import await_only

import milvusql.dbapi
from milvusql_sqlalchemy import asyncio_dbapi, reflection
from milvusql_sqlalchemy.compiler import MilvusDDLCompiler, MilvusSQLCompiler
from milvusql_sqlalchemy.padding import (
    PAD_VECTOR_FIELD,
    table_needs_pad_vector,
)

with contextlib.suppress(ImportError):
    # Registers `MilvusImpl` with Alembic (see that module's own
    # docstring for why importing it is what registers it) -- guarded
    # so this package works without Alembic installed at all; only
    # `MigrationContext.configure()` ever needs the registration to
    # already have happened, and that can't run before this dialect
    # module itself has been imported to resolve `create_engine(...)`
    # in the first place.
    from milvusql_sqlalchemy import alembic_impl as _alembic_impl  # noqa: F401

if t.TYPE_CHECKING:
    from sqlalchemy.engine import Connection as SAConnection
    from sqlalchemy.engine.interfaces import (
        DBAPIModule,
        IsolationLevel,
        ReflectedColumn,
        ReflectedForeignKeyConstraint,
        ReflectedIndex,
        ReflectedPrimaryKeyConstraint,
    )
    from sqlalchemy.engine.url import URL


def _unwrap(dbapi_connection: t.Any) -> t.Any:
    """The async dialect (``asyncio_dbapi.py``) hands out
    ``AsyncAdapt_dbapi_connection`` instances wrapping our real
    ``milvusql.aio.AsyncConnection`` -- not the connection itself.
    ``.driver_connection`` is that wrapper's own public unwrap point
    (confirmed directly against ``sqlalchemy.engine.interfaces
    .AdaptedConnection``); the sync dialect's ``dbapi_connection`` *is*
    already our raw ``milvusql.dbapi.Connection``, with no such
    attribute, so this is a no-op for it."""
    return getattr(dbapi_connection, "driver_connection", dbapi_connection)


def _raw_client(connection: SAConnection) -> t.Any:
    """The ``pymilvus.MilvusClient``/``AsyncMilvusClient`` behind a
    SQLAlchemy ``Connection`` -- reflection needs Milvus's own
    describe/list calls, which have no SQL-text equivalent to route
    through ``connection.execute()``."""
    dbapi_connection = connection.connection.dbapi_connection
    # Narrows `DBAPIConnection | None` for ty; a dead connection would
    # already have failed earlier in the call chain.
    assert dbapi_connection is not None  # noqa: S101  # nosec B101
    return _unwrap(dbapi_connection)._client


def _reflect(result: t.Any) -> t.Any:
    """``_raw_client()`` returns a plain ``MilvusClient`` under the
    sync dialect (an already-resolved value) or an ``AsyncMilvusClient``
    under ``MilvusDialect_aio`` (a coroutine) -- SQLAlchemy's own
    ``Dialect`` API declares every reflection method synchronous
    regardless of sync/async engine (confirmed directly:
    ``AsyncConnection`` exposes no async reflection methods of its
    own, only ``run_sync()``, which is the only way in to any of
    these). When the async engine is the one calling in, that already
    runs inside the greenlet ``run_sync()``/SQLAlchemy's own internal
    reflection bridge spawns, so ``await_only()`` -- the same bridge
    primitive ``sqlalchemy.connectors.asyncio`` itself uses -- is the
    correct, supported way to actually run the coroutine here, instead
    of it leaking back to the caller unawaited (confirmed directly:
    that used to surface as a bare, unhelpful ``TypeError`` deep in
    unrelated code)."""
    if inspect.isawaitable(result):
        return await_only(result)
    return result


class MilvusDialect(default.DefaultDialect):
    name = "milvusql"
    driver = "milvusql"

    statement_compiler = MilvusSQLCompiler
    ddl_compiler = MilvusDDLCompiler

    supports_statement_cache = True
    supports_native_boolean = True
    supports_alter = False
    supports_sane_rowcount = False
    supports_sane_multi_rowcount = False
    #: The DBAPI cursor's ``.lastrowid`` (``milvusql.dbapi.Cursor.execute``)
    #: is correctly populated from Milvus's server-assigned ``auto_id``
    #: after every INSERT -- ``True`` here is what makes SQLAlchemy
    #: actually read it back, both for Core's ``Result.inserted_primary_key``
    #: and (what the ORM's unit-of-work needs) a new object's identity key
    #: after ``Session.add()``/``commit()``.
    postfetch_lastrowid = True
    supports_default_values = False
    supports_empty_insert = False

    #: D6's Milvus consistency levels, exactly as ``pymilvus``'s
    #: ``ConsistencyLevel`` protobuf enum spells them (confirmed
    #: directly: a case-sensitive ``Enum.Value(...)`` lookup) --
    #: title-case, not the ``"READ COMMITTED"``-style
    #: all-``UPPERCASE`` names ``_assert_and_set_isolation_level``
    #: below exists specifically to stop SQLAlchemy from forcing this
    #: dialect's own values into.
    _CONSISTENCY_LEVELS = (
        "Strong",
        "Bounded",
        "Eventually",
        "Session",
        "Customized",
    )

    #: ``Table(..., milvusql_shards=2,
    #: milvusql_consistency_level="Bounded",
    #: milvusql_partition_key="category")`` / ``Index(...,
    #: milvusql_using="HNSW", milvusql_with={"metric_type": "COSINE",
    #: "M": 16})`` -- the standard SQLAlchemy mechanism for
    #: dialect-specific DDL options (same pattern as
    #: ``mysql_engine=...``), read back in ``compiler.py``.
    construct_arguments = [  # noqa: RUF012 -- ty rejects ClassVar here:
        # base `Dialect.construct_arguments` is itself declared as an
        # instance attribute, so annotating this as ClassVar is the
        # actual Liskov violation, not the other way around.
        (
            schema.Table,
            {"shards": None, "consistency_level": None, "partition_key": None},
        ),
        (schema.Index, {"using": None, "with": None}),
    ]

    @classmethod
    def import_dbapi(cls) -> DBAPIModule:
        # `milvusql.dbapi` satisfies this protocol at runtime (Error
        # hierarchy, connect(), paramstyle, ...) but ty's structural
        # check on the `Error` member is stricter than the protocol
        # needs to be here -- same kind of deliberate, understood cast
        # as `get_isolation_level`'s above.
        return t.cast("DBAPIModule", milvusql.dbapi)

    def create_connect_args(
        self, url: URL
    ) -> tuple[list[t.Any], dict[str, t.Any]]:
        if not url.host:
            # Milvus Lite: "milvusql:///relative.db" or
            # "milvusql:////abs/path.db" -- same convention as the
            # sqlite dialect, since a Lite target is a local file path,
            # not a host:port.
            kwargs: dict[str, t.Any] = {"uri": url.database or ""}
        else:
            scheme = "https" if url.query.get("secure") else "http"
            kwargs = {
                "uri": f"{scheme}://{url.host}:{url.port or 19530}",
                "db_name": url.database or "",
            }
        # "milvusql://user:password@host:port/db" -- Milvus's `token`
        # auth parameter is itself a "user:password" pair (confirmed
        # against pymilvus's docs), so a URL with both a username and
        # password reassembles them into one token rather than
        # forwarding the password alone and silently dropping the user.
        token = milvusql.dbapi.token_from_credentials(
            url.username, url.password
        )
        if token:
            kwargs["token"] = token
        return [], kwargs

    def do_rollback(self, dbapi_connection: t.Any) -> None:
        """No-op -- see the module docstring for why this deliberately
        does not call ``dbapi_connection.rollback()`` (D7)."""

    def get_isolation_level(self, dbapi_connection: t.Any) -> IsolationLevel:
        """D6's connection-level consistency-level fallback, riding
        SQLAlchemy's isolation-level extension point -- typed as
        ``IsolationLevel`` only to satisfy the base signature; the
        values that actually flow through here ("Bounded", "Strong",
        ...) are Milvus consistency levels, not one of the five SQL
        isolation levels that type enumerates. SQLAlchemy itself never
        validates the value against that ``Literal`` at runtime, only
        round-trips whatever ``set_isolation_level`` stored -- the cast
        below documents the deliberate, understood mismatch rather than
        silencing it."""
        level = _unwrap(dbapi_connection).consistency_level or "Bounded"
        return t.cast("IsolationLevel", level)

    def set_isolation_level(
        self, dbapi_connection: t.Any, level: IsolationLevel
    ) -> None:
        """A statement's own ``CONSISTENCY LEVEL`` clause still wins
        over this connection-level default -- that's enforced in
        ``milvusql.dbapi.Cursor.execute``, not here."""
        _unwrap(dbapi_connection).consistency_level = level

    def _assert_and_set_isolation_level(
        self, dbapi_conn: t.Any, level: IsolationLevel
    ) -> None:
        """Overridden, not just relied on via inheritance: the base
        ``DefaultDialect`` implementation unconditionally does
        ``level.replace("_", " ").upper()`` before validating/forwarding
        the value -- correct for the standard SQL isolation levels this
        hook was designed for ("READ COMMITTED", ...), but confirmed
        directly to corrupt every Milvus consistency level this dialect
        actually deals in (D6): ``"Strong"``/``"Bounded"`` become
        ``"STRONG"``/``"BOUNDED"``, and ``pymilvus``'s own
        ``get_consistency_level()`` does a case-sensitive lookup that
        then always raises ``InvalidConsistencyLevel`` -- for *every*
        later query too, since SQLAlchemy's connection-pool checkin
        reset (``reset_isolation_level``) round-trips through this same
        method using ``default_isolation_level``. This both replaces
        the base's case-mangling and does its own validation instead of
        ``_gen_allowed_isolation_levels``'s (which requires
        ``get_isolation_level_values()`` to return UPPERCASE names --
        the wrong convention for Milvus's, so this dialect does not
        implement that hook at all)."""
        if level not in self._CONSISTENCY_LEVELS:
            msg = (
                f"Invalid value {level!r} for isolation_level. Valid "
                f"consistency levels for {self.name!r} are "
                f"{', '.join(self._CONSISTENCY_LEVELS)}"
            )
            raise exc.ArgumentError(msg)
        self.set_isolation_level(dbapi_conn, level)

    def get_schema_names(
        self, connection: SAConnection, **kw: t.Any
    ) -> list[str]:
        return ["default"]

    def has_table(
        self,
        connection: SAConnection,
        table_name: str,
        schema: str | None = None,
        **kw: t.Any,
    ) -> bool:
        return table_name in self.get_table_names(connection, schema, **kw)

    def get_table_names(
        self, connection: SAConnection, schema: str | None = None, **kw: t.Any
    ) -> list[str]:
        return list(_reflect(_raw_client(connection).list_collections()))

    def get_view_names(
        self, connection: SAConnection, schema: str | None = None, **kw: t.Any
    ) -> list[str]:
        return []

    def get_columns(
        self,
        connection: SAConnection,
        table_name: str,
        schema: str | None = None,
        **kw: t.Any,
    ) -> list[ReflectedColumn]:
        description = _reflect(
            _raw_client(connection).describe_collection(table_name)
        )
        return reflection.columns_from_description(description)

    def get_pk_constraint(
        self,
        connection: SAConnection,
        table_name: str,
        schema: str | None = None,
        **kw: t.Any,
    ) -> ReflectedPrimaryKeyConstraint:
        description = _reflect(
            _raw_client(connection).describe_collection(table_name)
        )
        return reflection.pk_constraint_from_description(description)

    def get_foreign_keys(
        self,
        connection: SAConnection,
        table_name: str,
        schema: str | None = None,
        **kw: t.Any,
    ) -> list[ReflectedForeignKeyConstraint]:
        return []  # Milvus has no foreign keys -- honestly empty, not an error

    def get_indexes(
        self,
        connection: SAConnection,
        table_name: str,
        schema: str | None = None,
        **kw: t.Any,
    ) -> list[ReflectedIndex]:
        client = _raw_client(connection)
        return [
            reflection.index_from_describe(
                _reflect(client.describe_index(table_name, field_name))
            )
            for field_name in _reflect(client.list_indexes(table_name))
        ]


class MilvusDialect_aio(MilvusDialect):  # noqa: N801 -- matches upstream's own
    # naming for driver-suffixed dialect classes, e.g. SQLAlchemy's own
    # `SQLiteDialect_aiosqlite`, `PGDialect_asyncpg`.
    """``milvusql+aio`` -- for
    :func:`sqlalchemy.ext.asyncio.create_async_engine`.

    Everything (DDL rendering, reflection, ``construct_arguments``,
    the consistency-level fallback) is inherited unchanged from
    :class:`MilvusDialect`; only the driver name and the DBAPI module
    differ. See ``asyncio_dbapi.py`` for why no connection/cursor
    subclassing was needed to make ``milvusql.aio`` work with
    SQLAlchemy's generic async adapter."""

    driver = "aio"
    is_async = True
    # SQLAlchemy re-checks this per dialect *class*, not just via
    # inheritance -- confirmed directly: omitting it here (even though
    # `MilvusDialect` already sets it `True`) makes `create_async_engine`
    # warn "will not make use of SQL compilation caching" at import time.
    supports_statement_cache = True

    # `DefaultDialect.get_pool_class` just reads this attribute (falling
    # back to plain `QueuePool` if unset) -- it does NOT branch on
    # `is_async` itself, so every async dialect (`asyncmy`, `asyncpg`,
    # `aiosqlite`'s file-db case, ...) sets this explicitly. Without it,
    # `create_async_engine` raises "Pool class QueuePool cannot be used
    # with asyncio engine" (confirmed directly).
    poolclass = pool.AsyncAdaptedQueuePool

    @classmethod
    def import_dbapi(cls) -> DBAPIModule:
        # Same deliberate, understood cast as the sync dialect's
        # `import_dbapi` -- see its comment. No lazy import needed
        # here the way `aiosqlite`'s dialect needs one: `asyncio_dbapi`
        # only touches `milvusql`/`sqlalchemy`, both already hard
        # dependencies regardless of sync or async use, not an
        # optional third-party driver package.
        return t.cast("DBAPIModule", asyncio_dbapi.dbapi)

    def get_driver_connection(self, connection: t.Any) -> t.Any:
        """``DefaultDialect``'s own default (``return connection``) is
        correct for a normal DBAPI, but leaves an async dialect's own
        ``AdaptedConnection`` (``asyncio_dbapi.py``'s
        ``AsyncAdapt_dbapi_connection`` wrapper) unopened for anyone
        who calls ``AsyncConnection.get_raw_connection()`` and expects
        back what the docstring promises: the real driver connection,
        not the DBAPI-facing adapter -- confirmed directly (same
        pattern ``aiosqlite``'s own dialect overrides this hook for):
        without this, ``(await conn.get_raw_connection()).
        driver_connection`` still returns an ``AsyncAdapt_dbapi_connection``,
        not the ``milvusql.aio.AsyncConnection`` underneath it, and
        ``._client`` raises ``AttributeError``."""
        return connection._connection


@event.listens_for(schema.Table, "after_create")
def _load_pad_vector_table(
    target: schema.Table,
    connection: SAConnection,
    **kw: t.Any,  # noqa: ARG001 -- required by the `after_create` event signature
) -> None:
    """Registered globally on ``schema.Table``, not on any instance
    this package constructs itself -- deliberately, so this also
    catches a table Alembic creates internally (its own
    ``alembic_version`` bookkeeping table, chiefly) through this
    dialect without ever importing ``milvusql_sqlalchemy``'s own
    ``Table``/``MetaData`` objects. ``compiler.py``'s
    ``visit_create_table`` already spliced the hidden pad-vector
    column into the ``CREATE TABLE`` text itself (see ``padding.py``'s
    module docstring) -- Milvus still needs an index and a ``LOAD``
    before *any* query succeeds against a real server, not just vector
    search, the same requirement ``milvusql_django.schema.
    create_index_and_load`` exists for. Every dialect fires this event
    on every ``CREATE TABLE``, not just this one's -- the ``dialect.
    name`` guard is what keeps this a no-op for anyone using a
    different backend in the same process."""
    if connection.dialect.name != "milvusql":
        return
    if not table_needs_pad_vector(target):
        return
    table_name = target.name
    connection.exec_driver_sql(
        f'CREATE INDEX idx_{PAD_VECTOR_FIELD} ON "{table_name}" '
        f'("{PAD_VECTOR_FIELD}") USING HNSW '
        "WITH (metric_type='COSINE')"
    )
    connection.exec_driver_sql(f'LOAD TABLE "{table_name}"')


__all__ = ["MilvusDialect", "MilvusDialect_aio"]
