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
