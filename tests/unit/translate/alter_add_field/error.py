"""Error-path coverage for ``ALTER TABLE`` in
``translate.ast_to_pymilvus.build_call``.

sqlglot-milvus's grammar already rejects every ``ALTER`` action other
than ``ADD FIELD`` at *parse* time (``DROP``/``ALTER COLUMN``/``MODIFY``
all raise a ``ParseError``), so the only way to reach
``_build_alter_add_field``'s own defense-in-depth check is a synthetic
AST no real MilvusQL text could ever produce -- same reasoning as
``unsupported_statements/error.py``'s ``Unhandled`` node."""

from __future__ import annotations

import typing as t

import pytest
from pymilvus import MilvusClient
from sqlglot import exp
from sqlglot_milvus.expressions import AddField

import milvusql
from milvusql.translate.ast_to_pymilvus import build_call

pytestmark = [pytest.mark.unit, pytest.mark.translate]

#: Same deliberate, understood cast as ``tests.fixtures.translate``'s
#: ``_client`` -- ``build_call``'s signature wants an instance, but
#: nothing here ever reaches a method that reads ``self``.
_client = t.cast("MilvusClient", MilvusClient)


def test_a_non_add_field_action_raises_not_supported_error():
    ast = exp.Alter(
        this=exp.table_("items"),
        kind="TABLE",
        actions=[exp.Drop(this=exp.column("tag"), kind="COLUMN")],
    )
    with pytest.raises(
        milvusql.NotSupportedError, match="unsupported ALTER TABLE action"
    ):
        build_call(_client, ast, {})


def test_more_than_one_action_raises_not_supported_error():
    """``AddField`` is always the only action MilvusQL's own grammar
    produces (one ``ADD FIELD`` per statement); a synthetic AST with
    several actions is defended against the same way a foreign one
    would be."""
    field = exp.ColumnDef(
        this=exp.to_identifier("tag"), kind=exp.DataType.build("VARCHAR")
    )
    ast = exp.Alter(
        this=exp.table_("items"),
        kind="TABLE",
        actions=[AddField(this=field), AddField(this=field.copy())],
    )
    with pytest.raises(
        milvusql.NotSupportedError, match="unsupported ALTER TABLE action"
    ):
        build_call(_client, ast, {})
