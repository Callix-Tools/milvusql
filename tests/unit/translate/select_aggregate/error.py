"""Error-path coverage for ``translate.ast_to_pymilvus.build_call`` on
aggregate ``SELECT``s. ``GROUP BY`` is no longer among them -- it is
planned by ``translate.relational`` and covered in
``tests/unit/translate/select_group_by``; what stays here is the
single-collection reduction, the row ceiling that makes it honest, and
the search ordering it refuses to reduce over."""

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


class TestAggregateOverASearchIsRejected:
    """A query aggregate is exact over every matching row; an ANN
    ``search`` returns an approximate top-k. Asking for both in one
    statement used to drop the search silently and answer the exact
    question instead -- ``SELECT COUNT(*) FROM items ORDER BY embedding
    <=> :q LIMIT 10`` returned the whole collection's count."""

    def test_count_star_over_a_vector_search_raises(self, build_call_helper):
        with pytest.raises(
            milvusql.NotSupportedError, match="aggregate over a search"
        ):
            build_call_helper(
                "SELECT COUNT(*) FROM items "
                "ORDER BY embedding <=> :q LIMIT 10",
                {"q": [0.1, 0.2, 0.3, 0.4]},
            )

    def test_a_reduced_aggregate_over_a_vector_search_raises(
        self, build_call_helper
    ):
        """The reducing path fetched rows with no ``limit`` of its own
        and averaged over the collection, not over the ten hits."""
        with pytest.raises(
            milvusql.NotSupportedError, match="approximate top-k"
        ):
            build_call_helper(
                "SELECT AVG(price) FROM items "
                "ORDER BY embedding <-> :q LIMIT 10",
                {"q": [0.1, 0.2, 0.3, 0.4]},
            )

    def test_a_filter_does_not_make_it_acceptable(self, build_call_helper):
        """The ``WHERE`` narrows both calls alike; it is the ordering,
        not the predicate, that the exactness argument turns on."""
        with pytest.raises(milvusql.NotSupportedError):
            build_call_helper(
                "SELECT COUNT(*) FROM items WHERE price > 5 "
                "ORDER BY embedding <=> :q LIMIT 3",
                {"q": [0.1, 0.2, 0.3, 0.4]},
            )

    def test_a_full_text_score_ordering_is_rejected_too(
        self, build_call_helper
    ):
        """``BM25_SCORE`` runs as a ``search`` with ``metric_type=
        'BM25'`` exactly like a distance operator does, so it carries
        the same approximation."""
        with pytest.raises(milvusql.NotSupportedError):
            build_call_helper(
                "SELECT COUNT(*) FROM items "
                "ORDER BY BM25_SCORE(body_sparse, :q) LIMIT 5",
                {"q": "hello"},
            )

    def test_the_message_names_the_shape_that_works(self, build_call_helper):
        with pytest.raises(
            milvusql.NotSupportedError, match="in a subquery"
        ) as excinfo:
            build_call_helper(
                "SELECT COUNT(*) FROM items "
                "ORDER BY embedding <=> :q LIMIT 10",
                {"q": [0.1, 0.2, 0.3, 0.4]},
            )
        assert "exact" in str(excinfo.value)

    def test_the_subquery_spelling_the_message_names_really_works(
        self, build_call_helper
    ):
        """The two-step shape reaches the relational engine: the search
        runs as a real ``search`` bounded to its own ``LIMIT``, and the
        aggregate reduces over those hits rather than the collection."""
        call = build_call_helper(
            "SELECT COUNT(*) FROM ("
            "SELECT id FROM items ORDER BY embedding <=> :q LIMIT 10"
            ") AS hits",
            {"q": [0.1, 0.2, 0.3, 0.4]},
        )
        assert call.method == "search"
        assert call.kwargs["anns_field"] == "embedding"
        assert call.kwargs["limit"] == 10
        rows, _description, _rowcount, _lastrowid = call.postprocess(
            [[{"entity": {"id": i}} for i in range(4)]]
        )
        assert rows == [(4,)]

    def test_a_scalar_ordering_on_an_aggregate_still_compiles(
        self, build_call_helper
    ):
        """Only a search ordering is refused. A scalar ``ORDER BY``
        sorts the single aggregate row by a column that is not in the
        SELECT list -- it changes nothing, and rejecting it would break
        statements that work today."""
        call = build_call_helper(
            'SELECT COUNT(*) AS "n" FROM items ORDER BY price DESC LIMIT 5'
        )
        assert call.method == "query"
        assert call.kwargs["output_fields"] == ["count(*)"]
