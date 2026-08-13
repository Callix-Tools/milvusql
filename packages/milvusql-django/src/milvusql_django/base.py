"""``DatabaseWrapper`` -- the Django entry point (D11).

``CursorWrapper`` exists for one reason: Django's ORM always compiles
to ``%s``-positional placeholders no matter what the underlying DBAPI
wants (confirmed directly by reading ``django.db.backends.sqlite3
.base.SQLiteCursorWrapper``, which does the identical translation for
sqlite3's own "qmark" paramstyle) -- ``milvusql``'s DBAPI declares
``paramstyle = "named"`` (``:name``, Track A's MilvusQL bind syntax),
so this backend needs the same kind of bridge sqlite3's does, just
targeting a different paramstyle.

``do_rollback``/transactions: Milvus has no multi-statement rollback
(same reasoning as the SQLAlchemy dialect's ``do_rollback`` override --
see its module docstring for why that disagrees with
``milvusql.dbapi.Connection.rollback()`` on purpose). Django's
``BaseDatabaseWrapper._rollback`` is overridden here the same way, for
the same reason: Django's own connection-pool bookkeeping calls it as
routine cleanup, unrelated to a user asking to roll something back.
"""

from __future__ import annotations

import re
import typing as t

from django.db.backends.base.base import BaseDatabaseWrapper
from pymilvus.exceptions import MilvusException

import milvusql.dbapi as Database  # noqa: N812
from milvusql_django.client import DatabaseClient
from milvusql_django.creation import DatabaseCreation
from milvusql_django.features import DatabaseFeatures
from milvusql_django.introspection import DatabaseIntrospection
from milvusql_django.operations import DatabaseOperations
from milvusql_django.schema import (
    PAD_VECTOR_FIELD,
    PAD_VECTOR_VALUE,
    DatabaseSchemaEditor,
)

if t.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Matches a literal ``%s`` (Django's universal placeholder spelling),
#: skipping an escaped ``%%s`` the same way sqlite3's Django backend
#: does -- copied convention, not copied code.
_FORMAT_PLACEHOLDER = re.compile(r"(?<!%)%s")

#: Matches exactly the single-row ``INSERT`` text Django's own
#: ``SQLInsertCompiler`` emits (confirmed directly) -- deliberately
#: does not match a multi-row ``VALUES (...), (...)`` batch (from
#: ``bulk_create``): padding every row in a batch correctly needs a
#: full MilvusQL-aware rewrite, not a regex, and no model this backend
#: pads (see ``schema.PAD_VECTOR_FIELD``) is expected to see bulk
#: inserts -- it exists for Django's own single-row-at-a-time internal
#: bookkeeping models (``django_migrations``, chiefly). A batch insert
#: into a padded table falls through unpadded and fails with Milvus's
#: own honest "no float_vector data is set" error instead of silently
#: doing the wrong thing.
_INSERT_SINGLE_ROW_RE = re.compile(
    r'^INSERT INTO "(?P<table>[^"]+)" \((?P<columns>[^()]*)\) '
    r"VALUES \((?P<values>[^()]*)\)$"
)


class CursorWrapper:
    """Wraps ``milvusql.dbapi.Cursor``, translating Django's ``%s``/
    ``%(name)s`` placeholder text into MilvusQL's ``:name`` before
    handing it to the real cursor."""

    def __init__(
        self, cursor: Database.Cursor, wrapper: DatabaseWrapper
    ) -> None:
        self.cursor = cursor
        self._wrapper = wrapper

    def _pad_insert(
        self,
        query: str,
        params: Sequence[t.Any] | Mapping[str, t.Any] | None,
    ) -> tuple[str, Sequence[t.Any] | Mapping[str, t.Any] | None]:
        """Splice ``schema.PAD_VECTOR_FIELD``'s constant value into a
        single-row ``INSERT`` targeting a table ``create_model`` padded
        (see that module's docstring for why the padding exists at
        all) -- a no-op, zero-regex-match miss for every other
        ``INSERT`` and every other statement kind."""
        match = _INSERT_SINGLE_ROW_RE.match(query)
        if match is None or not self._wrapper._table_needs_pad_vector(
            match["table"]
        ):
            return query, params
        # S608: every interpolated piece is either a quoted identifier
        # already present in Django's own generated `query` text (the
        # table/column list `_INSERT_SINGLE_ROW_RE` just matched out of
        # it) or `PAD_VECTOR_FIELD`, a fixed constant this module
        # defines -- nothing here is external input.
        padded_query = (
            f'INSERT INTO "{match["table"]}" '  # noqa: S608  # nosec B608
            f'({match["columns"]}, "{PAD_VECTOR_FIELD}") '
            f"VALUES ({match['values']}, %s)"
        )
        padded_params = [*(params or ()), PAD_VECTOR_VALUE]
        return padded_query, padded_params

    def _convert(
        self,
        query: str,
        params: Sequence[t.Any] | Mapping[str, t.Any] | None,
    ) -> tuple[str, dict[str, t.Any]]:
        if params is None:
            return query, {}
        if isinstance(params, dict):
            named = query % {name: f":{name}" for name in params}
            return named, dict(params)
        names = [f"param{i}" for i in range(len(params))]
        it = iter(names)
        named = _FORMAT_PLACEHOLDER.sub(lambda _m: f":{next(it)}", query)
        return named, dict(zip(names, params, strict=True))

    def execute(
        self,
        query: str,
        params: Sequence[t.Any] | Mapping[str, t.Any] | None = None,
    ) -> t.Any:
        query, params = self._pad_insert(query, params)
        converted_query, bound = self._convert(query, params)
        return self.cursor.execute(converted_query, bound)

    def executemany(
        self,
        query: str,
        param_list: Sequence[Sequence[t.Any] | Mapping[str, t.Any]],
    ) -> t.Any:
        param_list = list(param_list)
        if not param_list:
            return None
        converted_query, _ = self._convert(query, param_list[0])
        bound_list = [self._convert(query, p)[1] for p in param_list]
        return self.cursor.executemany(converted_query, bound_list)

    def __getattr__(self, name: str) -> t.Any:
        return getattr(self.cursor, name)

    def __iter__(self) -> t.Any:
        return iter(self.cursor)


class DatabaseWrapper(BaseDatabaseWrapper):
    vendor = "milvus"
    display_name = "Milvus"

    Database = Database
    SchemaEditorClass = DatabaseSchemaEditor
    client_class = DatabaseClient
    creation_class = DatabaseCreation
    features_class = DatabaseFeatures
    introspection_class = DatabaseIntrospection
    ops_class = DatabaseOperations

    #: Django ``Field.get_internal_type()`` -> MilvusQL column type
    #: text, read by ``Field.db_type()``. Auto* fields all map to
    #: ``BIGINT``: Milvus's primary key must be ``INT64`` or
    #: ``VARCHAR`` (confirmed directly -- ``PrimaryKeyException`` for
    #: anything else), it has no INT32 auto-increment option to
    #: distinguish Django's Auto field size classes by.
    data_types = {  # noqa: RUF012 -- ty rejects ClassVar: base
        # `BaseDatabaseWrapper.data_types` is itself an instance
        # attribute, same situation as milvusql_sqlalchemy's
        # `construct_arguments`.
        "AutoField": "BIGINT",
        "BigAutoField": "BIGINT",
        "SmallAutoField": "BIGINT",
        "BooleanField": "BOOLEAN",
        "CharField": "VARCHAR(%(max_length)s)",
        "SlugField": "VARCHAR(%(max_length)s)",
        "EmailField": "VARCHAR(%(max_length)s)",
        "FloatField": "DOUBLE",
        "IntegerField": "INT",
        "BigIntegerField": "BIGINT",
        "SmallIntegerField": "INT",
        "PositiveIntegerField": "INT",
        "PositiveSmallIntegerField": "INT",
        "PositiveBigIntegerField": "BIGINT",
        "JSONField": "JSON",
        "TextField": "VARCHAR(65535)",
        # Milvus has no native date/time/decimal/UUID/binary column
        # type, so all of these round-trip as text -- exactly what
        # Django's own generic ``adapt_*field_value``/``get_prep_value``
        # hooks already hand this backend for them (confirmed directly:
        # ``BaseDatabaseOperations.adapt_datefield_value``/
        # ``adapt_datetimefield_value``/``adapt_timefield_value`` all
        # return ``str(value)`` unless a backend overrides them, and
        # ``UUIDField``/``GenericIPAddressField.get_prep_value()``
        # already return a plain string at the *field* level, before
        # any backend is even involved -- only ``DecimalField`` needed
        # a backend-level override, added on ``DatabaseOperations``
        # alongside these). Without an entry here at all, `db_type()`
        # returns ``None`` and ``CREATE TABLE``/``ALTER TABLE`` raise a
        # bare ``TypeError`` deep in ``schema.py`` -- confirmed
        # directly: this is exactly what broke Django's own
        # ``django_migrations`` bookkeeping table (``applied
        # DateTimeField``), which every ``migrate`` run creates first,
        # before a single user migration -- so this list needs every
        # built-in field `migrate` itself might reach, not just the
        # ones a hand-written model happens to use.
        "DateField": "VARCHAR(10)",
        "TimeField": "VARCHAR(15)",
        "DateTimeField": "VARCHAR(32)",
        "DurationField": "BIGINT",
        "DecimalField": "VARCHAR(%(max_digits)s)",
        "UUIDField": "VARCHAR(32)",
        "GenericIPAddressField": "VARCHAR(39)",
        # Unverified against a real Milvus deployment (D11: this
        # package is a first cut, same flag `DatabaseOperations.
        # sql_flush` carries) -- Django hands this backend a raw
        # `memoryview` for `BinaryField`, and nothing here encodes it
        # to the text a VARCHAR column can actually store.
        "BinaryField": "VARCHAR(65535)",
        # VectorField overrides db_type() directly (fields.py) --
        # never consults this dict.
    }

    #: Django's lookup name -> SQL operator template. Only what
    #: Milvus's filter DSL actually supports (verified against
    #: ``milvusql``'s own ``_render_filter``, which this backend's
    #: generated SQL text round-trips through): no ``LIKE`` wildcard
    #: translation, no regex.
    operators: t.ClassVar = {
        "exact": "= %s",
        "gt": "> %s",
        "gte": ">= %s",
        "lt": "< %s",
        "lte": "<= %s",
    }

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        #: See ``_table_needs_pad_vector`` -- one live ``describe_
        #: collection`` check per table per connection, not per query.
        self._pad_vector_cache: dict[str, bool] = {}

    def get_connection_params(self) -> dict[str, t.Any]:
        settings_dict = self.settings_dict
        name = settings_dict.get("NAME") or ""
        params: dict[str, t.Any] = {"uri": name}
        host = settings_dict.get("HOST")
        if host:
            scheme = (
                "https"
                if settings_dict.get("OPTIONS", {}).get("secure")
                else "http"
            )
            port = settings_dict.get("PORT") or 19530
            params["uri"] = f"{scheme}://{host}:{port}"
            params["db_name"] = name
        # DATABASES["default"]["USER"]/"PASSWORD" -- Milvus's `token`
        # auth parameter is itself a "user:password" pair (confirmed
        # against pymilvus's docs), so both settings reassemble into
        # one token rather than forwarding PASSWORD alone and silently
        # dropping USER.
        token = Database.token_from_credentials(
            settings_dict.get("USER"), settings_dict.get("PASSWORD")
        )
        if token:
            params["token"] = token
        params.update(settings_dict.get("OPTIONS", {}))
        return params

    def get_new_connection(self, conn_params: dict[str, t.Any]) -> t.Any:
        return Database.connect(**conn_params)

    def init_connection_state(self) -> None:
        pass

    def create_cursor(self, name: str | None = None) -> CursorWrapper:
        # Narrows for ty; Django's own `cursor()` already called
        # `ensure_connection()` before this runs.
        assert self.connection is not None  # noqa: S101  # nosec B101
        return CursorWrapper(self.connection.cursor(), self)

    def _table_needs_pad_vector(self, table: str) -> bool:
        """Whether ``table`` is one ``create_model`` gave a hidden
        ``PAD_VECTOR_FIELD`` to (see ``schema.py``'s module docstring)
        -- derived from Milvus's own live schema, not process memory,
        since the process that runs ``migrate`` and the process that
        later inserts into ``django_migrations`` are almost never the
        same one. Cached per connection: the answer for a given table
        cannot change without a ``DROP TABLE``/``CREATE TABLE``, which
        this backend never does behind a live connection's back."""
        cached = self._pad_vector_cache.get(table)
        if cached is not None:
            return cached
        assert self.connection is not None  # noqa: S101  # nosec B101
        try:
            description = self.connection._client.describe_collection(table)
        except MilvusException:
            # Doesn't exist (yet) or isn't reachable -- never claim a
            # table needs padding when we can't actually confirm it.
            return False
        needs_pad = any(
            field["name"] == PAD_VECTOR_FIELD
            for field in description["fields"]
        )
        self._pad_vector_cache[table] = needs_pad
        return needs_pad

    def _set_autocommit(self, autocommit: bool) -> None:
        """No-op: Milvus has no transactions to toggle (D7's own
        reasoning, one layer up)."""

    def _rollback(self) -> None:
        """No-op -- see the module docstring."""

    def is_usable(self) -> bool:
        return self.connection is not None and not self.connection.closed


__all__ = ["CursorWrapper", "DatabaseWrapper"]
