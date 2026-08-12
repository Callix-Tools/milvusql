"""Milvus schema introspection (``describe_collection``/``list_indexes``/
``describe_index``) -> SQLAlchemy's ``Reflected*`` TypedDict shapes.

Pure functions, no ``Connection``/``Dialect`` state, so they're testable
(by whoever writes those tests) without a live server: feed them the
same dicts ``pymilvus`` returns and check the shape that comes back.
"""

from __future__ import annotations

import typing as t

from pymilvus import DataType
from sqlalchemy import types as sa_types

from milvusql_sqlalchemy.types import VECTOR

if t.TYPE_CHECKING:
    from sqlalchemy.engine.interfaces import (
        ReflectedColumn,
        ReflectedIndex,
        ReflectedPrimaryKeyConstraint,
    )

#: Milvus field type -> SQLAlchemy column type. Only what phase-1 DDL
#: (D-decisions carried from ``milvusql`` core) actually emits;
#: anything else reflects as ``sqlalchemy.types.NullType`` rather than
#: guessing.
_TYPE_MAP: dict[int, t.Callable[[dict[str, t.Any]], sa_types.TypeEngine]] = {
    int(DataType.INT8): lambda _params: sa_types.SmallInteger(),
    int(DataType.INT16): lambda _params: sa_types.SmallInteger(),
    int(DataType.INT32): lambda _params: sa_types.Integer(),
    int(DataType.INT64): lambda _params: sa_types.BigInteger(),
    int(DataType.FLOAT): lambda _params: sa_types.Float(),
    int(DataType.DOUBLE): lambda _params: sa_types.Double(),
    int(DataType.BOOL): lambda _params: sa_types.Boolean(),
    int(DataType.JSON): lambda _params: sa_types.JSON(),
    int(DataType.VARCHAR): (
        lambda params: sa_types.String(length=params.get("max_length"))
    ),
}


def _column_type(field: dict[str, t.Any]) -> sa_types.TypeEngine:
    if int(field["type"]) == int(DataType.FLOAT_VECTOR):
        return VECTOR(dim=field.get("params", {}).get("dim"))
    factory = _TYPE_MAP.get(int(field["type"]))
    return factory(field.get("params", {})) if factory else sa_types.NullType()


def columns_from_description(
    description: dict[str, t.Any],
) -> list[ReflectedColumn]:
    return [
        {
            "name": field["name"],
            "type": _column_type(field),
            "nullable": not field.get("is_primary", False),
            "default": None,
            "autoincrement": field.get("auto_id", False),
        }
        for field in description["fields"]
    ]


def pk_constraint_from_description(
    description: dict[str, t.Any],
) -> ReflectedPrimaryKeyConstraint:
    pk_fields = [
        f["name"] for f in description["fields"] if f.get("is_primary")
    ]
    return {"constrained_columns": pk_fields, "name": None}


def index_from_describe(index_description: dict[str, t.Any]) -> ReflectedIndex:
    return {
        "name": index_description["index_name"],
        "column_names": [index_description["field_name"]],
        "unique": False,
        "dialect_options": {
            "milvusql_using": index_description.get("index_type"),
            "milvusql_with": {
                "metric_type": index_description.get("metric_type"),
            },
        },
    }


__all__ = [
    "columns_from_description",
    "index_from_describe",
    "pk_constraint_from_description",
]
