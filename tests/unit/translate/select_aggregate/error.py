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
    from a client-side row fetch, not server-side -- the fetch pages
    past Milvus's per-call row ceiling (it used to raise there), so the
    reduction runs over every matching row."""

    def test_hitting_the_row_ceiling_pages_instead_of_raising(
        self, build_call_helper, drive_chain
    ):
        """A read that comes back at the ceiling chains into a
        ``describe_collection`` (to learn the primary key) and re-reads
        page by page; the reduction then covers all pages' rows."""
        call = build_call_helper('SELECT SUM("id") AS "total" FROM items')
        first_page = [{"id": 1} for _ in range(DEFAULT_QUERY_LIMIT)]
        describe = {"fields": [{"name": "id", "is_primary": True}]}
        # Paged reads are cursor-ordered on the primary key.
        page_one = [{"id": i} for i in range(DEFAULT_QUERY_LIMIT)]
        page_two = [{"id": DEFAULT_QUERY_LIMIT}]
        calls, (rows, _d, _rc, _l) = drive_chain(
            call, [first_page, describe, page_one, page_two]
        )
        assert [c.method for c in calls] == [
            "query",
            "describe_collection",
            "query",
            "query",
        ]
        assert calls[2].kwargs["iterator"] == "True"
        assert calls[3].kwargs["filter"] == f"id > {DEFAULT_QUERY_LIMIT - 1}"
        total = sum(range(DEFAULT_QUERY_LIMIT)) + DEFAULT_QUERY_LIMIT
        assert rows == [(total,)]

    def test_an_unordered_page_raises_instead_of_losing_rows(
        self, build_call_helper, drive_chain
    ):
        """A server that ignores the ``iterator`` flag (Milvus Lite)
        returns pages in arbitrary order -- continuing would silently
        skip rows, so the read raises with the server named."""
        call = build_call_helper('SELECT SUM("id") AS "total" FROM items')
        first_page = [{"id": 1} for _ in range(DEFAULT_QUERY_LIMIT)]
        describe = {"fields": [{"name": "id", "is_primary": True}]}
        unordered = [
            {"id": 5},
            {"id": 3},
            *({"id": i + 10} for i in range(DEFAULT_QUERY_LIMIT - 2)),
        ]
        with pytest.raises(milvusql.NotSupportedError, match="ordered"):
            drive_chain(call, [first_page, describe, unordered])

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
