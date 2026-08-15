"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on
``DROP TABLE``: previously had no builder at all (fell through to the
generic "unsupported statement" error) -- found while exercising every
SQLAlchemy-documented feature of ``milvusql-sqlalchemy``, including the
everyday ``Table.drop()``/``metadata.drop_all()`` calls that compile to
this."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_maps_to_drop_collection(build_call_helper):
    call = build_call_helper("DROP TABLE items")
    assert call.method == "drop_collection"
    assert call.kwargs == {"collection_name": "items"}


def test_postprocess_returns_no_rows(build_call_helper):
    call = build_call_helper("DROP TABLE items")
    assert call.postprocess(object()) == ([], None, -1, None)


def test_if_exists_is_accepted_and_maps_the_same_as_plain_drop(
    build_call_helper,
):
    """``IF EXISTS`` parses fine (generic ``sqlglot`` ``Drop`` grammar
    already carries ``ast.args["exists"]``) but ``_build_drop_table``
    doesn't need to branch on it: Milvus's own ``drop_collection`` RPC
    is already a no-op, not an error, when the collection doesn't
    exist -- confirmed directly against a real server. So a table drop
    is unconditionally "drop if exists" already, with or without the
    caller spelling it out."""
    call = build_call_helper("DROP TABLE IF EXISTS items")
    assert call.method == "drop_collection"
    assert call.kwargs == {"collection_name": "items"}
