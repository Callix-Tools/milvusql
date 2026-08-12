"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on a
vector ``SELECT ... ORDER BY <-> ...``, dispatched to ``search``."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]

SQL = (
    "SELECT id, category FROM items WHERE category = :cat "
    "ORDER BY embedding <-> :q LIMIT 5 SEARCH PARAMS (ef_search=64)"
)


def test_dispatches_to_search_with_metric_and_knobs(build_call_helper):
    call = build_call_helper(SQL, {"cat": "book", "q": [0.1] * 8})
    assert call.method == "search"
    assert call.kwargs == {
        "collection_name": "items",
        "data": [[0.1] * 8],
        "anns_field": "embedding",
        "limit": 5,
        "output_fields": ["id", "category"],
        "search_params": {
            "metric_type": "L2",
            "params": {"ef_search": 64},
        },
        "filter": 'category == "book"',
    }


def test_consistency_level_clause_is_forwarded(build_call_helper):
    call = build_call_helper(
        "SELECT id FROM items ORDER BY embedding <-> :q LIMIT 5 "
        "CONSISTENCY LEVEL Eventually",
        {"q": [0.1] * 8},
    )
    assert call.kwargs["consistency_level"] == "Eventually"


def test_omits_consistency_level_when_statement_does_not_set_one(
    build_call_helper,
):
    call = build_call_helper(SQL, {"cat": "book", "q": [0.1] * 8})
    assert "consistency_level" not in call.kwargs


def test_search_postprocess_reads_the_top_level_hit_list(build_call_helper):
    call = build_call_helper(SQL, {"cat": "book", "q": [0.1] * 8})
    raw = [[{"entity": {"category": "book"}, "id": 1, "distance": 0.01}]]
    rows, _description, rowcount, _lastrowid = call.postprocess(raw)
    assert rows == [(1, "book")]
    assert rowcount == 1


def test_search_postprocess_handles_no_hits(build_call_helper):
    call = build_call_helper(SQL, {"cat": "book", "q": [0.1] * 8})
    assert call.postprocess([]) == (
        [],
        [
            ("id", None, None, None, None, None, True),
            ("category", None, None, None, None, None, True),
        ],
        0,
        None,
    )
