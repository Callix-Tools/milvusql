"""``DatabaseIntrospection`` -- reads Milvus's own schema description
(``describe_collection``) rather than probing with a ``SELECT``, same
choice ``milvusql_sqlalchemy.reflection`` made and for the same reason:
Milvus has a real schema API, no need to fake one through query text.
"""

from __future__ import annotations

import typing as t

from django.db.backends.base.introspection import (
    BaseDatabaseIntrospection,
    FieldInfo,
    TableInfo,
)
from pymilvus import DataType

#: Milvus field type -> Django field class name. Only what phase-1 DDL
#: (the design plan's D-decisions, carried through from ``milvusql``
#: core) actually emits; anything else falls back to ``TextField``
#: rather than guessing.
_TYPE_MAP: dict[int, str] = {
    int(DataType.INT8): "SmallIntegerField",
    int(DataType.INT16): "SmallIntegerField",
    int(DataType.INT32): "IntegerField",
    int(DataType.INT64): "BigIntegerField",
    int(DataType.FLOAT): "FloatField",
    int(DataType.DOUBLE): "FloatField",
    int(DataType.BOOL): "BooleanField",
    int(DataType.JSON): "JSONField",
    int(DataType.VARCHAR): "CharField",
    int(DataType.FLOAT_VECTOR): "VectorField",
}


class DatabaseIntrospection(BaseDatabaseIntrospection):
    data_types_reverse = _TYPE_MAP

    def get_field_type(self, data_type: int, description: FieldInfo) -> str:
        return _TYPE_MAP.get(data_type, "TextField")

    def get_table_list(self, cursor: t.Any) -> list[TableInfo]:
        client = self._client(cursor)
        return [TableInfo(name, "t") for name in client.list_collections()]

    def get_table_description(
        self, cursor: t.Any, table_name: str
    ) -> list[FieldInfo]:
        client = self._client(cursor)
        description = client.describe_collection(table_name)
        return [
            FieldInfo(
                name=field["name"],
                type_code=int(field["type"]),
                display_size=None,
                internal_size=field.get("params", {}).get("max_length"),
                precision=None,
                scale=None,
                null_ok=not field.get("is_primary", False),
                default=None,
                collation=None,
            )
            for field in description["fields"]
        ]

    def get_primary_key_columns(
        self, cursor: t.Any, table_name: str
    ) -> list[str]:
        client = self._client(cursor)
        description = client.describe_collection(table_name)
        return [
            f["name"] for f in description["fields"] if f.get("is_primary")
        ]

    def get_relations(
        self, cursor: t.Any, table_name: str
    ) -> dict[str, t.Any]:
        return {}  # Milvus has no foreign keys -- honestly empty

    def get_constraints(
        self, cursor: t.Any, table_name: str
    ) -> dict[str, t.Any]:
        return {}

    @staticmethod
    def _client(cursor: t.Any) -> t.Any:
        """The ``pymilvus.MilvusClient`` behind a wrapped cursor --
        introspection needs Milvus's own describe/list calls, which
        have no SQL-text equivalent to route through
        ``cursor.execute()`` (same reasoning as
        ``milvusql_sqlalchemy.dialect._raw_client``)."""
        raw = cursor.cursor if hasattr(cursor, "cursor") else cursor
        return raw.connection._client


__all__ = ["DatabaseIntrospection"]
