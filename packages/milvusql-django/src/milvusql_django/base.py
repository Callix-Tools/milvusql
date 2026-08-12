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

import milvusql.dbapi as Database  # noqa: N812
from milvusql_django.client import DatabaseClient
from milvusql_django.creation import DatabaseCreation
from milvusql_django.features import DatabaseFeatures
from milvusql_django.introspection import DatabaseIntrospection
from milvusql_django.operations import DatabaseOperations
from milvusql_django.schema import DatabaseSchemaEditor

if t.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Matches a literal ``%s`` (Django's universal placeholder spelling),
#: skipping an escaped ``%%s`` the same way sqlite3's Django backend
#: does -- copied convention, not copied code.
_FORMAT_PLACEHOLDER = re.compile(r"(?<!%)%s")


class CursorWrapper:
    """Wraps ``milvusql.dbapi.Cursor``, translating Django's ``%s``/
    ``%(name)s`` placeholder text into MilvusQL's ``:name`` before
    handing it to the real cursor."""

    def __init__(self, cursor: Database.Cursor) -> None:
        self.cursor = cursor

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
        return CursorWrapper(self.connection.cursor())

    def _set_autocommit(self, autocommit: bool) -> None:
        """No-op: Milvus has no transactions to toggle (D7's own
        reasoning, one layer up)."""

    def _rollback(self) -> None:
        """No-op -- see the module docstring."""

    def is_usable(self) -> bool:
        return self.connection is not None and not self.connection.closed


__all__ = ["CursorWrapper", "DatabaseWrapper"]
