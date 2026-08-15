"""Unit coverage for window functions.

The shape that matters against a vector database is "top-k per group":
``ROW_NUMBER() OVER (PARTITION BY category ORDER BY distance)`` over the
hits of a search, then a filter on the rank. Milvus has no window
concept at all, so the whole clause is evaluated client-side over the
fetched rows."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]

ROWS = [
    {"id": 1, "category": "book", "price": 10.0},
    {"id": 2, "category": "book", "price": 30.0},
    {"id": 3, "category": "film", "price": 20.0},
]


class TestPlan:
    def test_a_window_query_still_reads_once(self, build_call_helper):
        call = build_call_helper(
            "SELECT id, ROW_NUMBER() OVER (ORDER BY price) AS r FROM items"
        )
        assert call.method == "query"
        assert call.then is None
        assert sorted(call.kwargs["output_fields"]) == ["id", "price"]

    def test_the_partition_and_order_columns_are_fetched(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT id, RANK() OVER (PARTITION BY category ORDER BY price) "
            "AS r FROM items"
        )
        assert sorted(call.kwargs["output_fields"]) == [
            "category",
            "id",
            "price",
        ]


class TestRanking:
    def test_row_number_restarts_per_partition(self, build_call_helper):
        call = build_call_helper(
            "SELECT id, ROW_NUMBER() OVER "
            "(PARTITION BY category ORDER BY price DESC) AS r FROM items"
        )
        rows, _desc, _rc, _l = call.postprocess(ROWS)
        assert rows == [(1, 2), (2, 1), (3, 1)]

    def test_rank_leaves_gaps_after_a_tie(self, build_call_helper):
        call = build_call_helper(
            "SELECT id, RANK() OVER (ORDER BY price) AS r FROM items"
        )
        tied = [
            {"id": 1, "price": 10.0},
            {"id": 2, "price": 10.0},
            {"id": 3, "price": 20.0},
        ]
        rows, _desc, _rc, _l = call.postprocess(tied)
        assert rows == [(1, 1), (2, 1), (3, 3)]

    def test_dense_rank_does_not(self, build_call_helper):
        call = build_call_helper(
            "SELECT id, DENSE_RANK() OVER (ORDER BY price) AS r FROM items"
        )
        tied = [
            {"id": 1, "price": 10.0},
            {"id": 2, "price": 10.0},
            {"id": 3, "price": 20.0},
        ]
        rows, _desc, _rc, _l = call.postprocess(tied)
        assert rows == [(1, 1), (2, 1), (3, 2)]


class TestAggregateWindows:
    def test_a_sum_over_a_partition_broadcasts_back(self, build_call_helper):
        """Unlike ``GROUP BY``, this keeps one row per input row."""
        call = build_call_helper(
            "SELECT id, SUM(price) OVER (PARTITION BY category) AS total "
            "FROM items"
        )
        rows, _desc, rowcount, _l = call.postprocess(ROWS)
        assert rows == [(1, 40.0), (2, 40.0), (3, 20.0)]
        assert rowcount == 3

    def test_count_over_a_partition(self, build_call_helper):
        call = build_call_helper(
            "SELECT id, COUNT(*) OVER (PARTITION BY category) AS n FROM items"
        )
        rows, _desc, _rc, _l = call.postprocess(ROWS)
        assert rows == [(1, 2), (2, 2), (3, 1)]

    def test_no_partition_means_one_partition(self, build_call_helper):
        call = build_call_helper(
            "SELECT id, SUM(price) OVER () AS total FROM items"
        )
        rows, _desc, _rc, _l = call.postprocess(ROWS)
        assert rows == [(1, 60.0), (2, 60.0), (3, 60.0)]


class TestRejected:
    """These raise when the window is evaluated rather than when the
    statement is planned -- the same place the ungrouped-column and
    unsupported-aggregate errors surface, since all three are properties
    of the expression tree the evaluator walks."""

    def test_a_frame_clause_is_rejected(self, build_call_helper):
        with pytest.raises(milvusql.NotSupportedError, match="frame"):
            build_call_helper(
                "SELECT SUM(price) OVER "
                "(ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s "
                "FROM items"
            ).postprocess(ROWS)

    def test_an_unsupported_window_function_is_named(self, build_call_helper):
        """``LAG`` is an ``AggFunc`` in sqlglot but has no reduction
        behind it -- the message says which function is missing."""
        with pytest.raises(milvusql.NotSupportedError, match="LAG"):
            build_call_helper(
                "SELECT LAG(price) OVER (ORDER BY id) AS p FROM items"
            ).postprocess(ROWS)

    def test_ranking_without_an_order_is_rejected(self, build_call_helper):
        with pytest.raises(milvusql.NotSupportedError, match="ORDER BY"):
            build_call_helper(
                "SELECT RANK() OVER (PARTITION BY category) AS r FROM items"
            ).postprocess(ROWS)

    def test_ranking_on_several_keys_is_rejected(self, build_call_helper):
        with pytest.raises(milvusql.NotSupportedError, match="one key"):
            build_call_helper(
                "SELECT RANK() OVER (ORDER BY price, id) AS r FROM items"
            ).postprocess(ROWS)
