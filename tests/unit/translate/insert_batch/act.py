"""Unit coverage for ``translate.ast_to_pymilvus.build_batch_call`` --
``executemany()``'s entry point: a parsed statement plus every one of
its bind-parameter sets -> a single batched ``Call``, or ``None`` when
Milvus has no batched primitive for the statement shape."""

from __future__ import annotations

import pytest
import sqlglot

from milvusql.translate.ast_to_pymilvus import build_batch_call

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def _parse(sql: str):
    return sqlglot.parse_one(sql, read="milvus")


def test_insert_merges_every_parameter_set_into_one_call():
    ast = _parse("INSERT INTO items (id, category) VALUES (:id, :cat)")
    call = build_batch_call(
        ast,
        [
            {"id": 1, "cat": "a"},
            {"id": 2, "cat": "b"},
            {"id": 3, "cat": "c"},
        ],
    )
    assert call is not None
    assert call.method == "insert"
    assert call.kwargs == {
        "collection_name": "items",
        "data": [
            {"id": 1, "category": "a"},
            {"id": 2, "category": "b"},
            {"id": 3, "category": "c"},
        ],
    }


def test_insert_batches_multi_row_values_across_parameter_sets_too():
    """A multi-``VALUES`` ``INSERT`` combined with several parameter
    sets -- each parameter set contributes every row its own ``VALUES``
    clause names, not just one."""
    ast = _parse(
        "INSERT INTO items (id, category) VALUES (:id1, :cat1), (:id2, :cat2)"
    )
    call = build_batch_call(
        ast,
        [
            {"id1": 1, "cat1": "a", "id2": 2, "cat2": "b"},
            {"id1": 3, "cat1": "c", "id2": 4, "cat2": "d"},
        ],
    )
    assert call is not None
    assert call.kwargs["data"] == [
        {"id": 1, "category": "a"},
        {"id": 2, "category": "b"},
        {"id": 3, "category": "c"},
        {"id": 4, "category": "d"},
    ]


def test_insert_postprocess_is_the_same_reducer_a_single_insert_uses():
    """``_insert_result`` needs no batch-specific variant: it already
    reduces a multi-row ``insert()`` response to one rowcount/lastrowid
    pair regardless of whether the rows came from one multi-VALUES
    statement or many parameter sets."""
    ast = _parse("INSERT INTO items (category) VALUES (:cat)")
    call = build_batch_call(ast, [{"cat": "a"}, {"cat": "b"}])
    assert call is not None
    rows, description, rowcount, lastrowid = call.postprocess(
        {"insert_count": 2, "ids": [10, 11]}
    )
    assert rows == []
    assert description is None
    assert rowcount == 2
    assert lastrowid == 11


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE items SET category = :cat WHERE id = :id",
        "DELETE FROM items WHERE category = :cat",
        "SELECT id FROM items WHERE category = :cat",
    ],
    ids=["update", "delete", "select"],
)
def test_statements_with_no_batched_primitive_return_none(sql):
    """Milvus has no batched primitive for these -- the caller falls
    back to one ``build_call`` per parameter set."""
    ast = _parse(sql)
    assert build_batch_call(ast, [{"cat": "a", "id": 1}]) is None
