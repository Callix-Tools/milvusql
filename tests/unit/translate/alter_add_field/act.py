"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on
``ALTER TABLE ... ADD FIELD`` / ``ADD COLUMN``: parsed MilvusQL AST ->
the ``Call`` (method, kwargs, postprocess) it dispatches to.

sqlglot-milvus's own grammar rejects every other ``ALTER`` action
(``DROP``/``ALTER COLUMN``/``MODIFY``) with a ``ParseError`` at parse
time -- Milvus genuinely cannot perform them -- so an add-a-field
action, in either its Milvus-native ``ADD FIELD`` spelling (wrapped in
``AddField``) or standard SQL's ``ADD COLUMN`` spelling (a bare
``ColumnDef``), is the only ``ALTER TABLE`` shape that ever reaches
``build_call`` through the real grammar."""

from __future__ import annotations

import pytest
from pymilvus import DataType

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_scalar_field_dispatches_to_add_collection_field(build_call_helper):
    call = build_call_helper("ALTER TABLE items ADD FIELD tag VARCHAR(32)")
    assert call.method == "add_collection_field"
    assert call.kwargs == {
        "collection_name": "items",
        "field_name": "tag",
        "data_type": DataType.VARCHAR,
        "nullable": True,
        "max_length": 32,
    }


def test_added_field_is_always_nullable(build_call_helper):
    """Existing rows have no value to backfill for a brand-new column,
    and a nullable vector field is the one shape pymilvus itself
    enforces (``ParamError`` otherwise) -- setting it unconditionally
    is correct for every column kind, not just vectors."""
    call = build_call_helper("ALTER TABLE items ADD FIELD score DOUBLE")
    assert call.kwargs["nullable"] is True
    assert call.kwargs["data_type"] == DataType.DOUBLE


def test_standard_sql_add_column_dispatches_identically(build_call_helper):
    """Standard SQL's ``ADD COLUMN`` (e.g. as rendered by Alembic's
    default ``add_column()``) parses to a bare ``ColumnDef`` action
    rather than Milvus's own ``AddField``-wrapped one, but must build
    the identical ``add_collection_field`` call."""
    call = build_call_helper("ALTER TABLE items ADD COLUMN tag VARCHAR(32)")
    assert call.method == "add_collection_field"
    assert call.kwargs == {
        "collection_name": "items",
        "field_name": "tag",
        "data_type": DataType.VARCHAR,
        "nullable": True,
        "max_length": 32,
    }


def test_postprocess_returns_no_rows(build_call_helper):
    call = build_call_helper("ALTER TABLE items ADD FIELD tag VARCHAR(32)")
    rows, description, rowcount, lastrowid = call.postprocess(None)
    assert rows == []
    assert description is None
    assert rowcount == -1
    assert lastrowid is None
