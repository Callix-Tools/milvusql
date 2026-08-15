"""Unit coverage for reading past Milvus's per-call row ceiling: the
primary-key-cursor page loop behind every unbounded read (plain
``SELECT`` with no ``LIMIT``, client-side ``ORDER BY``, aggregates,
``UPDATE``'s read, relational scans)."""

from __future__ import annotations

import pytest

import milvusql
from milvusql.translate._common import DEFAULT_QUERY_LIMIT

pytestmark = [pytest.mark.unit, pytest.mark.translate]

DESCRIBE = {"fields": [{"name": "id", "is_primary": True}]}


def _page(start: int, count: int) -> list[dict]:
    return [{"id": index} for index in range(start, start + count)]


class TestUnderTheCeiling:
    def test_a_small_result_is_exactly_one_rpc(self, build_call_helper):
        """The overwhelmingly common case must stay byte-identical to
        the single call made before pagination existed."""
        call = build_call_helper("SELECT id FROM items")
        assert call.method == "query"
        assert call.kwargs["limit"] == DEFAULT_QUERY_LIMIT
        assert "iterator" not in call.kwargs
        assert call.then([{"id": 1}]) is None
        rows, _d, rowcount, _l = call.postprocess([{"id": 1}])
        assert rows == [(1,)]
        assert rowcount == 1

    def test_an_explicit_limit_never_pages(self, build_call_helper):
        call = build_call_helper("SELECT id FROM items LIMIT 5")
        assert call.kwargs["limit"] == 5
        assert call.then is None


class TestPastTheCeiling:
    def test_pages_are_cursor_chained_on_the_primary_key(
        self, build_call_helper, drive_chain
    ):
        call = build_call_helper("SELECT id FROM items WHERE id >= 0")
        first = _page(0, DEFAULT_QUERY_LIMIT)
        remainder = _page(DEFAULT_QUERY_LIMIT, 3)
        calls, (_rows, _d, rowcount, _l) = drive_chain(
            call, [first, DESCRIBE, first, remainder]
        )
        assert [c.method for c in calls] == [
            "query",
            "describe_collection",
            "query",
            "query",
        ]
        # Every page speaks the same iterator protocol pymilvus's own
        # QueryIterator does, and carries the original filter AND the
        # cursor.
        assert calls[2].kwargs["iterator"] == "True"
        assert calls[2].kwargs["filter"] == "id >= 0"
        assert calls[3].kwargs["filter"] == (
            f"(id >= 0) and (id > {DEFAULT_QUERY_LIMIT - 1})"
        )
        assert rowcount == DEFAULT_QUERY_LIMIT + 3

    def test_the_primary_key_is_added_to_output_fields_for_paging(
        self, build_call_helper, drive_chain
    ):
        call = build_call_helper("SELECT name FROM items")
        first = [{"name": f"n{index}"} for index in range(DEFAULT_QUERY_LIMIT)]
        pages = [
            {"id": index, "name": f"n{index}"}
            for index in range(DEFAULT_QUERY_LIMIT)
        ]
        calls, (rows, _d, rowcount, _l) = drive_chain(
            call, [first, DESCRIBE, pages, [{"id": 99999, "name": "last"}]]
        )
        assert calls[2].kwargs["output_fields"] == ["name", "id"]
        # The extra key column never leaks into the result rows.
        assert rows[-1] == ("last",)
        assert rowcount == DEFAULT_QUERY_LIMIT + 1

    def test_a_string_primary_key_cursor_is_quoted(
        self, build_call_helper, drive_chain
    ):
        call = build_call_helper("SELECT pk FROM items")
        describe = {"fields": [{"name": "pk", "is_primary": True}]}
        first = [
            {"pk": f"k{index:06d}"} for index in range(DEFAULT_QUERY_LIMIT)
        ]
        calls, _result = drive_chain(
            call, [first, describe, first, [{"pk": "z"}]]
        )
        assert (
            calls[3].kwargs["filter"]
            == f'pk > "k{DEFAULT_QUERY_LIMIT - 1:06d}"'
        )

    def test_an_unordered_page_raises_instead_of_losing_rows(
        self, build_call_helper, drive_chain
    ):
        """Milvus Lite ignores the iterator flag and returns arbitrary
        order -- continuing would skip rows silently, so the read stops
        with the reason named."""
        call = build_call_helper("SELECT id FROM items")
        unordered = [{"id": 5}, {"id": 3}, *_page(10, DEFAULT_QUERY_LIMIT - 2)]
        with pytest.raises(milvusql.NotSupportedError, match="ordered"):
            drive_chain(
                call, [_page(0, DEFAULT_QUERY_LIMIT), DESCRIBE, unordered]
            )
