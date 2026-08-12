"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on
``INSERT``."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_single_row_maps_columns_to_a_data_dict(build_call_helper):
    call = build_call_helper(
        "INSERT INTO items (embedding, category) VALUES (:emb, :cat)",
        {"emb": [0.1] * 8, "cat": "book"},
    )
    assert call.method == "insert"
    assert call.kwargs == {
        "collection_name": "items",
        "data": [{"embedding": [0.1] * 8, "category": "book"}],
    }


def test_multi_row_values_become_one_data_dict_per_row(build_call_helper):
    call = build_call_helper(
        "INSERT INTO items (id, category) VALUES (1, 'a'), (2, 'b')"
    )
    assert call.kwargs["data"] == [
        {"id": 1, "category": "a"},
        {"id": 2, "category": "b"},
    ]


def test_postprocess_reports_insert_count_and_last_row_id(build_call_helper):
    call = build_call_helper(
        "INSERT INTO items (category) VALUES (:cat)", {"cat": "book"}
    )
    assert call.postprocess({"insert_count": 1, "ids": [42]}) == (
        [],
        None,
        1,
        42,
    )


def test_postprocess_handles_no_ids_for_non_auto_id_collections(
    build_call_helper,
):
    call = build_call_helper(
        "INSERT INTO items (id, category) VALUES (:id, :cat)",
        {"id": 1, "cat": "book"},
    )
    assert call.postprocess({"insert_count": 1, "ids": []}) == (
        [],
        None,
        1,
        None,
    )
