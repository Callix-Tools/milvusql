"""Unit coverage for ``translate.relational`` on ``GROUP BY``: one
Milvus read, then the reduction client-side.

Milvus computes no grouped aggregate at all (``count(*)`` server-side is
the ungrouped exception, and stays on the single-call path), so the
whole clause is planned as "fetch the referenced columns, then reduce"
-- with the row-ceiling guard making truncation an error rather than a
wrong number."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]

ROWS = [
    {"category": "book", "price": 10.0, "stock": 1},
    {"category": "book", "price": 20.0, "stock": None},
    {"category": "film", "price": 30.0, "stock": 3},
]


class TestPlan:
    def test_fetches_only_the_grouped_and_aggregated_columns(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT category, AVG(price) FROM items GROUP BY category"
        )
        assert call.method == "query"
        assert call.kwargs["collection_name"] == "items"
        assert sorted(call.kwargs["output_fields"]) == ["category", "price"]
        assert call.kwargs["limit"] == 16384

    def test_a_where_filter_still_goes_to_milvus(self, build_call_helper):
        call = build_call_helper(
            "SELECT category, COUNT(*) FROM items WHERE stock > 0 "
            "GROUP BY category"
        )
        assert call.kwargs["filter"] == "stock > 0"

    def test_it_is_a_single_rpc(self, build_call_helper):
        """One collection, one read -- grouping adds no round trip."""
        call = build_call_helper(
            "SELECT category, COUNT(*) FROM items GROUP BY category"
        )
        assert call.then is None


class TestReduction:
    def test_count_and_avg_per_group(self, build_call_helper):
        call = build_call_helper(
            "SELECT category, COUNT(*) AS n, AVG(price) AS avg_price "
            "FROM items GROUP BY category"
        )
        rows, desc, rowcount, _lastrowid = call.postprocess(ROWS)
        assert rows == [("book", 2, 15.0), ("film", 1, 30.0)]
        assert [column[0] for column in desc] == ["category", "n", "avg_price"]
        assert rowcount == 2

    def test_count_of_a_column_ignores_nulls(self, build_call_helper):
        """``COUNT(<column>)`` counts non-null values; ``COUNT(*)``
        counts rows. The ``book`` group has two rows but one stock."""
        call = build_call_helper(
            "SELECT category, COUNT(stock) AS n FROM items GROUP BY category"
        )
        rows, _desc, _rowcount, _lastrowid = call.postprocess(ROWS)
        assert rows == [("book", 1), ("film", 1)]

    def test_count_distinct_ignores_nulls_too(self, build_call_helper):
        call = build_call_helper(
            "SELECT COUNT(DISTINCT category) AS n, COUNT(DISTINCT stock) AS s "
            "FROM items GROUP BY 1 = 1"
        )
        rows, _desc, _rowcount, _lastrowid = call.postprocess(ROWS)
        assert rows == [(2, 2)]

    def test_min_max_sum(self, build_call_helper):
        call = build_call_helper(
            "SELECT category, MIN(price) AS lo, MAX(price) AS hi, "
            "SUM(price) AS total FROM items GROUP BY category"
        )
        rows, _desc, _rowcount, _lastrowid = call.postprocess(ROWS)
        assert rows == [("book", 10.0, 20.0, 30.0), ("film", 30.0, 30.0, 30.0)]

    def test_having_filters_groups_not_rows(self, build_call_helper):
        call = build_call_helper(
            "SELECT category, COUNT(*) AS n FROM items "
            "GROUP BY category HAVING COUNT(*) > 1"
        )
        rows, _desc, _rowcount, _lastrowid = call.postprocess(ROWS)
        assert rows == [("book", 2)]

    def test_order_by_an_aggregate_alias_then_limit(self, build_call_helper):
        call = build_call_helper(
            "SELECT category, MAX(price) AS hi FROM items "
            "GROUP BY category ORDER BY hi DESC LIMIT 1"
        )
        rows, _desc, _rowcount, _lastrowid = call.postprocess(ROWS)
        assert rows == [("film", 30.0)]

    def test_order_by_the_aggregate_itself(self, build_call_helper):
        """``ORDER BY COUNT(*)`` names the reduction, not a label -- it
        resolves to the same computed column the SELECT list uses."""
        call = build_call_helper(
            "SELECT category FROM items GROUP BY category "
            "ORDER BY COUNT(*) DESC"
        )
        rows, _desc, _rowcount, _lastrowid = call.postprocess(ROWS)
        assert rows == [("book",), ("film",)]

    def test_an_expression_over_two_aggregates(self, build_call_helper):
        """``SUM(price) / COUNT(*)`` cannot be expressed as one
        reduction: each aggregate is computed, then the arithmetic runs
        over the reduced rows."""
        call = build_call_helper(
            "SELECT category, SUM(price) / COUNT(*) AS mean FROM items "
            "GROUP BY category"
        )
        rows, _desc, _rowcount, _lastrowid = call.postprocess(ROWS)
        assert rows == [("book", 15.0), ("film", 30.0)]

    def test_grouping_by_several_keys(self, build_call_helper):
        call = build_call_helper(
            "SELECT category, stock, COUNT(*) AS n FROM items "
            "GROUP BY category, stock"
        )
        rows, _desc, _rowcount, _lastrowid = call.postprocess(ROWS)
        assert rows == [
            ("book", 1, 1),
            ("book", None, 1),
            ("film", 3, 1),
        ]

    def test_an_empty_collection_produces_no_groups(self, build_call_helper):
        call = build_call_helper(
            "SELECT category, COUNT(*) AS n FROM items GROUP BY category"
        )
        rows, _desc, rowcount, _lastrowid = call.postprocess([])
        assert rows == []
        assert rowcount == 0
