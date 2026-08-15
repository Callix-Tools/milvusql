"""Error-path coverage for ``GROUP BY``: what SQL itself rejects, and
what this engine cannot compute."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestUngroupedColumns:
    def test_a_column_that_is_neither_grouped_nor_aggregated_is_rejected(
        self, build_call_helper
    ):
        """It has no single value per group. SQL rejects it; picking one
        row's value silently -- which a naive translation would do --
        returns a number nobody asked for."""
        call = build_call_helper(
            "SELECT category, name, COUNT(*) FROM items GROUP BY category"
        )
        with pytest.raises(milvusql.ProgrammingError, match="GROUP BY"):
            call.postprocess([{"category": "book", "name": "a"}])

    def test_the_grouping_key_itself_is_fine(self, build_call_helper):
        call = build_call_helper(
            "SELECT category, COUNT(*) FROM items GROUP BY category"
        )
        rows, _desc, _rowcount, _lastrowid = call.postprocess(
            [{"category": "book"}]
        )
        assert rows == [("book", 1)]


class TestUnsupportedAggregates:
    def test_an_unknown_aggregate_is_rejected(self, build_call_helper):
        """Milvus computes none of these itself and this engine reduces
        only the five standard ones -- anything else is a real gap, said
        out loud rather than dropped from the SELECT list."""
        call = build_call_helper(
            "SELECT category, STDDEV(price) FROM items GROUP BY category"
        )
        with pytest.raises(milvusql.NotSupportedError, match="unsupported"):
            call.postprocess([{"category": "book", "price": 1.0}])
