"""Error-path coverage for ``translate.ast_to_pymilvus.build_call`` on
``GROUP BY`` -- a shape Milvus cannot execute (no server-side grouping,
nothing downstream reduces per-group), so it must raise a clean
``NotSupportedError`` instead of silently running ungrouped."""

from __future__ import annotations

import pytest

import milvusql
from milvusql.translate.ast_to_pymilvus import _DEFAULT_QUERY_LIMIT

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestGroupByIsRejected:
    def test_group_by_raises_not_supported_error(self, build_call_helper):
        """``_is_aggregate_select`` already returns ``False`` whenever
        ``GROUP BY`` is present, so this used to fall through to the
        plain-select path and execute as an ungrouped ``query()`` --
        one row per matching entity instead of one row per group, with
        no error at all."""
        with pytest.raises(milvusql.NotSupportedError, match="GROUP BY"):
            build_call_helper(
                "SELECT category, COUNT(*) FROM items GROUP BY category"
            )

    def test_group_by_is_rejected_even_with_a_bare_column_select(
        self, build_call_helper
    ):
        with pytest.raises(milvusql.NotSupportedError, match="GROUP BY"):
            build_call_helper("SELECT category FROM items GROUP BY category")


class TestRowCeilingIsDetected:
    """Milvus computes ``SUM``/``AVG``/``MIN``/``MAX``/``COUNT(<col>)``
    from a client-side row fetch, not server-side -- a WHERE filter
    matching more than Milvus's own per-call row ceiling would
    otherwise reduce over an arbitrary truncated subset and report a
    wrong number with no error at all."""

    def test_hitting_the_row_ceiling_raises_instead_of_reducing_a_partial_view(
        self, build_call_helper
    ):
        call = build_call_helper('SELECT SUM("id") AS "total" FROM items')
        raw = [{"id": i} for i in range(_DEFAULT_QUERY_LIMIT)]
        with pytest.raises(milvusql.NotSupportedError, match="aggregate"):
            call.postprocess(raw)

    def test_one_row_under_the_ceiling_still_reduces_normally(
        self, build_call_helper
    ):
        call = build_call_helper('SELECT SUM("id") AS "total" FROM items')
        raw = [{"id": 1} for _ in range(_DEFAULT_QUERY_LIMIT - 1)]
        rows, _description, rowcount, _lastrowid = call.postprocess(raw)
        assert rows == [(_DEFAULT_QUERY_LIMIT - 1,)]
        assert rowcount == 1

    def test_a_pure_count_star_is_exempt_from_the_ceiling(
        self, build_call_helper
    ):
        """``COUNT(*)`` alone is computed server-side (see
        ``_count_star_rows``) -- it never fetches rows at all, so it
        has no ceiling to hit regardless of how many rows match."""
        call = build_call_helper('SELECT COUNT(*) AS "n" FROM items')
        assert call.kwargs.get("limit") is None
        rows, _description, _rowcount, _lastrowid = call.postprocess(
            [{"count(*)": _DEFAULT_QUERY_LIMIT + 1}]
        )
        assert rows == [(_DEFAULT_QUERY_LIMIT + 1,)]
