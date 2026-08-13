"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on a
plain (non-vector) ``SELECT ... ORDER BY <scalar column>`` -- dispatched
to ``query``, sorted client-side in postprocess (Milvus's ``query()``
RPC has no ordering concept of its own)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_scalar_order_by_still_dispatches_to_query(build_call_helper):
    call = build_call_helper(
        "SELECT id, price FROM items ORDER BY price DESC LIMIT 5"
    )
    assert call.method == "query"
    # Fetches every matching row (up to Milvus's own ceiling), not just
    # the requested LIMIT -- sorting after truncating would return the
    # wrong rows.
    assert call.kwargs["limit"] == 16384
    assert call.kwargs["output_fields"] == ["id", "price"]


def test_order_by_column_not_in_the_select_list_is_still_fetched(
    build_call_helper,
):
    call = build_call_helper(
        "SELECT id FROM items ORDER BY price DESC LIMIT 5"
    )
    assert call.kwargs["output_fields"] == ["id", "price"]


def test_postprocess_sorts_descending_then_applies_the_real_limit(
    build_call_helper,
):
    call = build_call_helper(
        "SELECT id, price FROM items ORDER BY price DESC LIMIT 2"
    )
    raw = [
        {"id": 1, "price": 9.99},
        {"id": 2, "price": 29.99},
        {"id": 3, "price": 19.99},
    ]
    rows, _description, rowcount, _lastrowid = call.postprocess(raw)
    assert rows == [(2, 29.99), (3, 19.99)]
    assert rowcount == 2


def test_postprocess_sorts_ascending_by_default(build_call_helper):
    call = build_call_helper("SELECT id, price FROM items ORDER BY price")
    raw = [{"id": 1, "price": 30}, {"id": 2, "price": 10}]
    rows, _description, _rowcount, _lastrowid = call.postprocess(raw)
    assert rows == [(2, 10), (1, 30)]


def test_multi_key_order_by_breaks_ties_with_the_second_key(
    build_call_helper,
):
    call = build_call_helper(
        "SELECT id, category, price FROM items ORDER BY category, price DESC"
    )
    raw = [
        {"id": 1, "category": "b", "price": 1},
        {"id": 2, "category": "a", "price": 5},
        {"id": 3, "category": "a", "price": 10},
    ]
    rows, _description, _rowcount, _lastrowid = call.postprocess(raw)
    assert rows == [(3, "a", 10), (2, "a", 5), (1, "b", 1)]
