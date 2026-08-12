"""Unit coverage for ``reflection.py``'s pure functions -- fed literal
dicts shaped like real ``describe_collection()``/``describe_index()``
output (verified directly against a live Milvus Lite client before
writing these), per the module's own docstring: no ``Connection``/
``Dialect`` state, testable without a live server."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.reflection import (
    columns_from_description,
    index_from_describe,
    pk_constraint_from_description,
)
from milvusql_sqlalchemy.types import VECTOR
from pymilvus import DataType
from sqlalchemy import types as sa_types

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]

#: Shaped exactly like ``MilvusClient.describe_collection()``'s real
#: return value for a collection built from ``id BIGINT PRIMARY KEY
#: AUTO_INCREMENT, embedding VECTOR(8), category VARCHAR(64)``.
_ITEMS_DESCRIPTION = {
    "collection_name": "items",
    "auto_id": True,
    "num_shards": 1,
    "fields": [
        {
            "field_id": 0,
            "name": "id",
            "description": "",
            "type": int(DataType.INT64),
            "params": {},
            "is_primary": True,
            "auto_id": True,
        },
        {
            "field_id": 0,
            "name": "embedding",
            "description": "",
            "type": int(DataType.FLOAT_VECTOR),
            "params": {"dim": 8},
        },
        {
            "field_id": 0,
            "name": "category",
            "description": "",
            "type": int(DataType.VARCHAR),
            "params": {"max_length": 64},
        },
    ],
}


class TestColumnsFromDescription:
    def test_primary_key_column_is_not_nullable(self):
        columns = columns_from_description(_ITEMS_DESCRIPTION)
        id_col = next(c for c in columns if c["name"] == "id")
        assert id_col["nullable"] is False
        assert id_col["autoincrement"] is True
        assert isinstance(id_col["type"], sa_types.BigInteger)

    def test_non_primary_columns_are_nullable_and_not_autoincrement(self):
        columns = columns_from_description(_ITEMS_DESCRIPTION)
        category_col = next(c for c in columns if c["name"] == "category")
        assert category_col["nullable"] is True
        assert category_col["autoincrement"] is False
        assert category_col["default"] is None

    def test_vector_field_becomes_a_vector_type_with_its_dim(self):
        columns = columns_from_description(_ITEMS_DESCRIPTION)
        embedding_col = next(c for c in columns if c["name"] == "embedding")
        assert isinstance(embedding_col["type"], VECTOR)
        assert embedding_col["type"].dim == 8

    def test_varchar_field_carries_its_max_length(self):
        columns = columns_from_description(_ITEMS_DESCRIPTION)
        category_col = next(c for c in columns if c["name"] == "category")
        assert isinstance(category_col["type"], sa_types.String)
        assert category_col["type"].length == 64

    @pytest.mark.parametrize(
        ("data_type", "expected"),
        [
            (DataType.INT8, sa_types.SmallInteger),
            (DataType.INT16, sa_types.SmallInteger),
            (DataType.INT32, sa_types.Integer),
            (DataType.FLOAT, sa_types.Float),
            (DataType.DOUBLE, sa_types.Double),
            (DataType.BOOL, sa_types.Boolean),
            (DataType.JSON, sa_types.JSON),
        ],
    )
    def test_scalar_types_map_to_their_sqlalchemy_equivalent(
        self, data_type, expected
    ):
        description = {
            "fields": [{"name": "f", "type": int(data_type), "params": {}}]
        }
        columns = columns_from_description(description)
        assert isinstance(columns[0]["type"], expected)


class TestPkConstraintFromDescription:
    def test_reports_the_single_primary_key_column(self):
        pk = pk_constraint_from_description(_ITEMS_DESCRIPTION)
        assert pk == {"constrained_columns": ["id"], "name": None}

    def test_no_primary_key_field_yields_an_empty_column_list(self):
        description = {"fields": [{"name": "f", "type": int(DataType.INT64)}]}
        pk = pk_constraint_from_description(description)
        assert pk == {"constrained_columns": [], "name": None}


class TestIndexFromDescribe:
    def test_maps_describe_index_output_to_a_reflected_index(self):
        # Shaped like a real ``describe_index()`` call -- observed
        # directly that Milvus Lite's reflected `index_name` can equal
        # the field name rather than the name CREATE INDEX gave it,
        # so this only asserts the shape this function produces from
        # whatever `index_name` it's handed, not that round-trip.
        index_description = {
            "field_name": "embedding",
            "index_name": "embedding",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "indexed_rows": 0,
            "pending_index_rows": 0,
            "state": "Finished",
            "total_rows": 0,
        }
        assert index_from_describe(index_description) == {
            "name": "embedding",
            "column_names": ["embedding"],
            "unique": False,
            "dialect_options": {
                "milvusql_using": "HNSW",
                "milvusql_with": {"metric_type": "COSINE"},
            },
        }
