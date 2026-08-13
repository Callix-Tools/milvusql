"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on
aggregate ``SELECT``s -- the shape Django's ``QuerySet.count()``/
``aggregate()`` always compile to (one row, no ``GROUP BY``)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestCountStar:
    """A pure ``COUNT(*)`` (Django's ``.count()``) is computed
    server-side via Milvus's own ``count(*)`` output field -- no row
    fetch, no ceiling."""

    SQL = 'SELECT COUNT(*) AS "__count" FROM items WHERE active'

    def test_dispatches_to_query_with_count_star_output_field(
        self, build_call_helper
    ):
        call = build_call_helper(self.SQL)
        assert call.method == "query"
        assert call.kwargs == {
            "collection_name": "items",
            "output_fields": ["count(*)"],
            "filter": "active",
        }

    def test_postprocess_reads_the_native_count(self, build_call_helper):
        call = build_call_helper(self.SQL)
        rows, _description, rowcount, _lastrowid = call.postprocess(
            [{"count(*)": 42}]
        )
        assert rows == [(42,)]
        assert rowcount == 1

    def test_postprocess_handles_an_empty_collection(self, build_call_helper):
        call = build_call_helper(self.SQL)
        rows, _description, _rowcount, _lastrowid = call.postprocess([])
        assert rows == [(0,)]


class TestGenericAggregates:
    """``SUM``/``AVG``/``MIN``/``MAX``/``COUNT(<column>)`` have no
    Milvus-native equivalent -- fetch the referenced columns and reduce
    them client-side in postprocess."""

    def test_sum_fetches_only_the_referenced_column(self, build_call_helper):
        call = build_call_helper('SELECT SUM("stock") AS "total" FROM items')
        assert call.method == "query"
        assert call.kwargs["output_fields"] == ["stock"]
        assert call.kwargs["limit"] == 16384

    def test_sum_postprocess_adds_up_the_column(self, build_call_helper):
        call = build_call_helper('SELECT SUM("stock") AS "total" FROM items')
        rows, _description, rowcount, _lastrowid = call.postprocess(
            [{"stock": 5}, {"stock": 10}]
        )
        assert rows == [(15,)]
        assert rowcount == 1

    def test_sum_of_no_matching_rows_is_zero_not_none(self, build_call_helper):
        call = build_call_helper('SELECT SUM("stock") AS "total" FROM items')
        rows, _description, _rowcount, _lastrowid = call.postprocess([])
        assert rows == [(0,)]

    def test_avg_min_max_reduce_correctly(self, build_call_helper):
        call = build_call_helper(
            'SELECT AVG("stock") AS "a", MIN("stock") AS "m", '
            'MAX("stock") AS "x" FROM items'
        )
        raw = [{"stock": 1}, {"stock": 2}, {"stock": 3}]
        rows, _description, _rowcount, _lastrowid = call.postprocess(raw)
        assert rows == [(2, 1, 3)]

    def test_count_of_a_real_column_counts_non_null_values(
        self, build_call_helper
    ):
        call = build_call_helper('SELECT COUNT("stock") AS "n" FROM items')
        rows, _description, _rowcount, _lastrowid = call.postprocess(
            [{"stock": 1}, {"stock": None}, {"stock": 3}]
        )
        assert rows == [(2,)]

    def test_multiple_aggregates_share_one_fetch(self, build_call_helper):
        call = build_call_helper(
            'SELECT SUM("stock") AS "total", COUNT("id") AS "n" FROM items'
        )
        assert sorted(call.kwargs["output_fields"]) == ["id", "stock"]
        rows, _description, _rowcount, _lastrowid = call.postprocess(
            [{"stock": 5, "id": 1}, {"stock": 10, "id": 2}]
        )
        assert rows == [(15, 2)]
