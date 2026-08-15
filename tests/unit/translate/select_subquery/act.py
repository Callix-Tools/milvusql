"""Unit coverage for ``translate.relational`` on subqueries: an inner
``SELECT`` is one more Milvus read in the same chain, evaluated before
the predicate or the outer query that consumes it.

Note what is *not* here: ``SELECT ... FROM (SELECT ... FROM t WHERE ...)
AS x`` with nothing else to it never reaches this engine at all --
``ast_to_pymilvus._flatten_trivial_subquery`` folds that shape (the one
SQLAlchemy's legacy ``Query.count()`` wraps everything in) back into a
single call. What lands here is the rest: a subquery in a predicate, and
a derived table the outer query actually groups or joins."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]

ITEMS = [
    {"id": 1, "cat_id": 10, "price": 10.0},
    {"id": 2, "cat_id": 11, "price": 30.0},
    {"id": 3, "cat_id": 12, "price": 50.0},
]
CATS = [{"id": 10}, {"id": 11}]


class TestInPredicate:
    SQL = (
        "SELECT id FROM items WHERE cat_id IN "
        "(SELECT id FROM cats WHERE active)"
    )

    def test_the_inner_select_is_its_own_read(self, build_call_helper):
        """The subquery's collection is read too -- and first, since it
        carries the filter."""
        call = build_call_helper(self.SQL)
        assert call.kwargs["collection_name"] == "cats"
        assert call.kwargs["filter"] == "active == true"
        assert call.kwargs["output_fields"] == ["id"]

    def test_the_outer_rows_are_filtered_by_the_inner_ones(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, rowcount, _l) = drive_chain(
            build_call_helper(self.SQL), [CATS, ITEMS]
        )
        assert rows == [(1,), (2,)]
        assert rowcount == 2

    def test_not_in_negates_it(self, build_call_helper, drive_chain):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT id FROM items WHERE cat_id NOT IN "
                "(SELECT id FROM cats WHERE active)"
            ),
            [CATS, ITEMS],
        )
        assert rows == [(3,)]

    def test_an_empty_inner_result_matches_nothing(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(self.SQL), [[], ITEMS]
        )
        assert rows == []


class TestScalarSubquery:
    def test_a_one_row_subquery_is_used_as_a_value(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT id FROM items WHERE price > "
                "(SELECT AVG(price) FROM items)"
            ),
            [ITEMS, ITEMS],
        )
        # AVG(price) over the three rows is 30.0
        assert rows == [(3,)]

    def test_more_than_one_row_is_an_error(
        self, build_call_helper, drive_chain
    ):
        with pytest.raises(milvusql.ProgrammingError, match="more than one"):
            drive_chain(
                build_call_helper(
                    "SELECT id FROM items WHERE cat_id > (SELECT id FROM cats)"
                ),
                [CATS, ITEMS],
            )


class TestExists:
    def test_a_non_empty_subquery_keeps_every_row(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT id FROM items WHERE EXISTS (SELECT id FROM cats)"
            ),
            [CATS, ITEMS],
        )
        assert rows == [(1,), (2,), (3,)]

    def test_an_empty_subquery_drops_every_row(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT id FROM items WHERE EXISTS (SELECT id FROM cats)"
            ),
            [[], ITEMS],
        )
        assert rows == []


class TestDerivedTable:
    SQL = (
        "SELECT t.cat_id, COUNT(*) AS n FROM "
        "(SELECT cat_id, price FROM items WHERE price > 5) AS t "
        "GROUP BY t.cat_id"
    )

    def test_the_inner_filter_is_pushed_to_milvus(self, build_call_helper):
        call = build_call_helper(self.SQL)
        assert call.kwargs["collection_name"] == "items"
        assert call.kwargs["filter"] == "price > 5"

    def test_the_outer_query_groups_the_inner_result(self, build_call_helper):
        call = build_call_helper(self.SQL)
        rows, desc, _rowcount, _lastrowid = call.postprocess(ITEMS)
        assert rows == [(10, 1), (11, 1), (12, 1)]
        assert [column[0] for column in desc] == ["cat_id", "n"]

    def test_an_inner_limit_is_honored(self, build_call_helper):
        """A ``LIMIT`` inside the subquery changes which rows reach the
        outer query, so it is applied to the inner result -- not folded
        away, and not confused with the outer query's own."""
        call = build_call_helper(
            "SELECT t.cat_id, COUNT(*) AS n FROM "
            "(SELECT cat_id FROM items LIMIT 2) AS t GROUP BY t.cat_id"
        )
        rows, _desc, _rowcount, _lastrowid = call.postprocess(ITEMS)
        assert rows == [(10, 1), (11, 1)]


class TestSubqueryColumnsStayInTheSubquery:
    """A subquery's columns belong to the subquery's collection.

    With one source in scope every bare column resolves to it, so the
    inner ``title``/``id`` used to be added to the *outer* collection's
    ``output_fields`` -- which Milvus Lite quietly ignores and a real
    server rejects outright ("field title not exist")."""

    SQL = (
        "SELECT price FROM joined_items WHERE cat_id IN "
        "(SELECT id FROM categories WHERE title = :t)"
    )

    def test_the_outer_read_asks_only_for_its_own_fields(
        self, build_call_helper, drive_chain
    ):
        calls, _result = drive_chain(
            build_call_helper(self.SQL, {"t": "book"}),
            [[{"id": 10}], [{"price": 1.0, "cat_id": 10}]],
        )
        outer = next(
            call
            for call in calls
            if call.kwargs["collection_name"] == "joined_items"
        )
        assert sorted(outer.kwargs["output_fields"]) == ["cat_id", "price"]

    def test_the_inner_read_asks_only_for_its_own_fields(
        self, build_call_helper
    ):
        call = build_call_helper(self.SQL, {"t": "book"})
        assert call.kwargs["collection_name"] == "categories"
        assert call.kwargs["output_fields"] == ["id"]
        assert call.kwargs["filter"] == 'title == "book"'

    def test_the_rows_still_come_out_right(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(self.SQL, {"t": "book"}),
            [
                [{"id": 10}],
                [{"price": 1.0, "cat_id": 10}, {"price": 2.0, "cat_id": 99}],
            ],
        )
        assert rows == [(1.0,)]
