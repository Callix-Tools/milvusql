"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on
``UPDATE`` -- a two-step ``Call``: read the matching rows in full, then
``then`` chains into an ``upsert`` of the merged rows (Milvus has no
partial-row update RPC)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]

SQL = 'UPDATE items SET "stock" = :n WHERE "name" = :who'


def test_first_step_is_a_full_row_read_of_the_matching_rows(
    build_call_helper,
):
    call = build_call_helper(SQL, {"n": 5, "who": "Widget"})
    assert call.method == "query"
    assert call.kwargs == {
        "collection_name": "items",
        "output_fields": ["*"],
        "limit": 16384,
        "filter": 'name == "Widget"',
    }
    assert call.then is not None


def test_no_matching_rows_stops_the_chain_with_rowcount_zero(
    build_call_helper,
):
    call = build_call_helper(SQL, {"n": 5, "who": "Widget"})
    assert call.then([]) is None
    rows, _description, rowcount, _lastrowid = call.postprocess([])
    assert rows == []
    assert rowcount == 0


def test_matching_rows_chain_into_an_upsert_of_the_merged_rows(
    build_call_helper,
):
    call = build_call_helper(SQL, {"n": 5, "who": "Widget"})
    fetched = [{"id": 1, "name": "Widget", "stock": 1, "embedding": [0.1]}]
    next_call = call.then(fetched)
    assert next_call.method == "upsert"
    assert next_call.kwargs == {
        "collection_name": "items",
        "data": [{"id": 1, "name": "Widget", "stock": 5, "embedding": [0.1]}],
    }


def test_upsert_postprocess_reports_the_upsert_count(build_call_helper):
    """The upsert's `then` runs before its `postprocess` (the cursor's
    own order) -- it accumulates the chunk's count and, with every row
    written, ends the chain; `postprocess` then reports the total."""
    call = build_call_helper(SQL, {"n": 5, "who": "Widget"})
    next_call = call.then([{"id": 1, "name": "Widget", "stock": 1}])
    assert next_call.then({"upsert_count": 1}) is None
    _rows, _description, rowcount, _lastrowid = next_call.postprocess(
        {"upsert_count": 1}
    )
    assert rowcount == 1


def test_update_with_no_where_clause_matches_every_row(build_call_helper):
    call = build_call_helper('UPDATE items SET "active" = :v', {"v": False})
    assert "filter" not in call.kwargs
