"""``DatabaseIntrospection`` reads Milvus's own schema description
API (``describe_collection``) rather than probing with a ``SELECT`` --
unit coverage against a stub ``MilvusClient`` reached through the same
``cursor.cursor.connection._client`` chain a real wrapped cursor
exposes (``base.CursorWrapper.cursor`` is the inner ``milvusql.dbapi
.Cursor``, whose ``.connection._client`` is the ``MilvusClient``)."""

from __future__ import annotations

import typing as t
from types import SimpleNamespace

import pytest
from django.db.backends.base.introspection import FieldInfo
from milvusql_django.introspection import DatabaseIntrospection
from pymilvus import DataType

pytestmark = [pytest.mark.unit, pytest.mark.django]

#: `get_field_type` is a plain lookup that reads neither `self` nor
#: `description` (confirmed by reading `introspection.py``) -- these
#: stand in for the types its signature declares without building a
#: real instance/`FieldInfo`.
_NO_INTROSPECTION = t.cast(DatabaseIntrospection, None)
_NO_FIELD_INFO = t.cast(FieldInfo, None)


class TestGetFieldType:
    def test_a_known_data_type_maps_to_its_django_field_class_name(self):
        assert (
            DatabaseIntrospection.get_field_type(
                _NO_INTROSPECTION, int(DataType.INT64), _NO_FIELD_INFO
            )
            == "BigIntegerField"
        )

    def test_an_unmapped_data_type_falls_back_to_text_field(self):
        assert (
            DatabaseIntrospection.get_field_type(
                _NO_INTROSPECTION, 9999, _NO_FIELD_INFO
            )
            == "TextField"
        )


class _FakeClient:
    def list_collections(self):
        return ["items", "other"]

    def describe_collection(self, name):
        return {
            "fields": [
                {"name": "id", "type": DataType.INT64, "is_primary": True},
                {
                    "name": "category",
                    "type": DataType.VARCHAR,
                    "params": {"max_length": 64},
                    "is_primary": False,
                },
            ]
        }


@pytest.fixture
def stub_cursor():
    """Shaped like a real wrapped cursor: ``.cursor`` is the inner
    DBAPI cursor, whose ``.connection._client`` is the ``MilvusClient``
    introspection needs -- these calls have no SQL-text equivalent to
    route through ``cursor.execute()``."""
    client = _FakeClient()
    return SimpleNamespace(
        cursor=SimpleNamespace(connection=SimpleNamespace(_client=client))
    )


class TestSchemaReads:
    def test_get_table_list_returns_every_collection_name(self, stub_cursor):
        introspection = DatabaseIntrospection(connection=None)
        tables = introspection.get_table_list(stub_cursor)
        assert [(table.name, table.type) for table in tables] == [
            ("items", "t"),
            ("other", "t"),
        ]

    def test_get_table_description_maps_every_field(self, stub_cursor):
        introspection = DatabaseIntrospection(connection=None)
        fields = introspection.get_table_description(stub_cursor, "items")
        assert [
            (f.name, f.type_code, f.null_ok, f.internal_size) for f in fields
        ] == [
            ("id", int(DataType.INT64), False, None),
            ("category", int(DataType.VARCHAR), True, 64),
        ]

    def test_get_primary_key_columns_returns_only_primary_fields(
        self, stub_cursor
    ):
        introspection = DatabaseIntrospection(connection=None)
        assert introspection.get_primary_key_columns(stub_cursor, "items") == [
            "id"
        ]

    def test_get_relations_is_honestly_empty(self, stub_cursor):
        introspection = DatabaseIntrospection(connection=None)
        assert introspection.get_relations(stub_cursor, "items") == {}

    def test_get_constraints_is_honestly_empty(self, stub_cursor):
        introspection = DatabaseIntrospection(connection=None)
        assert introspection.get_constraints(stub_cursor, "items") == {}
