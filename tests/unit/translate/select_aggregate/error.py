"""Error-path coverage for ``translate.ast_to_pymilvus.build_call`` on
aggregate ``SELECT``s. ``GROUP BY`` is no longer among them -- it is
planned by ``translate.relational`` and covered in
``tests/unit/translate/select_group_by``; what stays here is the
single-collection reduction and the row ceiling that makes it honest."""

from __future__ import annotations

import pytest

import milvusql
from milvusql.translate._common import DEFAULT_QUERY_LIMIT

pytestmark = [pytest.mark.unit, pytest.mark.translate]


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
        raw = [{"id": i} for i in range(DEFAULT_QUERY_LIMIT)]
        with pytest.raises(milvusql.NotSupportedError, match="aggregate"):
            call.postprocess(raw)

    def test_one_row_under_the_ceiling_still_reduces_normally(
        self, build_call_helper
    ):
        call = build_call_helper('SELECT SUM("id") AS "total" FROM items')
        raw = [{"id": 1} for _ in range(DEFAULT_QUERY_LIMIT - 1)]
        rows, _description, rowcount, _lastrowid = call.postprocess(raw)
        assert rows == [(DEFAULT_QUERY_LIMIT - 1,)]
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
            [{"count(*)": DEFAULT_QUERY_LIMIT + 1}]
        )
        assert rows == [(DEFAULT_QUERY_LIMIT + 1,)]
