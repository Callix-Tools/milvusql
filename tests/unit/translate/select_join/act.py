"""Unit coverage for ``translate.relational`` on ``JOIN``: one Milvus
read per collection, then the join itself evaluated client-side.

These drive the ``Call`` chain over canned pymilvus results, so they
assert both halves of the contract -- what each RPC asks Milvus for
(collection, fields, filter, and the join keys pushed into the second
one) and what the combined rows come back as."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]

SQL = (
    "SELECT i.id, i.name, c.title FROM items AS i "
    "JOIN cats AS c ON i.cat_id = c.id WHERE i.price > 20"
)
ITEMS = [
    {"id": 1, "name": "a", "cat_id": 10, "price": 30.0},
    {"id": 2, "name": "b", "cat_id": 11, "price": 40.0},
]
CATS = [{"id": 10, "title": "books"}, {"id": 11, "title": "films"}]


class TestPlansOneReadPerCollection:
    def test_first_call_reads_the_filtered_collection(self, build_call_helper):
        """The ``WHERE`` conjunct names one collection, so it is pushed
        into that collection's own filter rather than evaluated
        client-side -- and that scan runs first, since a filtered read
        is the more selective one to drive the join from."""
        call = build_call_helper(SQL)
        assert call.method == "query"
        assert call.kwargs["collection_name"] == "items"
        assert call.kwargs["filter"] == "price > 20"
        assert set(call.kwargs["output_fields"]) == {
            "id",
            "name",
            "cat_id",
        }

    def test_second_call_reads_the_other_collection(
        self, build_call_helper, drive_chain
    ):
        calls, _result = drive_chain(build_call_helper(SQL), [ITEMS, CATS])
        assert [c.kwargs["collection_name"] for c in calls] == [
            "items",
            "cats",
        ]
        assert set(calls[1].kwargs["output_fields"]) == {"id", "title"}

    def test_only_referenced_fields_are_requested(self, build_call_helper):
        """Projection pushdown: ``price`` is fetched because the filter
        that mentions it was pushed down, but nothing else the statement
        never names is."""
        call = build_call_helper(
            "SELECT i.id FROM items AS i JOIN cats AS c ON i.cat_id = c.id"
        )
        assert sorted(call.kwargs["output_fields"]) == ["cat_id", "id"]


class TestKeyPushdown:
    def test_join_keys_narrow_the_second_read(
        self, build_call_helper, drive_chain
    ):
        """The whole point of reading the driving side first: the second
        collection is asked only for the keys that can actually match,
        so joining a small result against a large collection stays
        small."""
        calls, _result = drive_chain(build_call_helper(SQL), [ITEMS, CATS])
        assert calls[1].kwargs["filter"] == "id in [10, 11]"

    def test_an_empty_driving_side_short_circuits(
        self, build_call_helper, drive_chain
    ):
        """Nothing matched the first collection, so no row of the second
        can survive an inner join -- say that in the filter instead of
        reading the collection to throw all of it away."""
        calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(SQL), [[], []]
        )
        assert calls[1].kwargs["filter"] == "false"
        assert rows == []

    def test_a_full_join_pushes_nothing_down(
        self, build_call_helper, drive_chain
    ):
        """A ``FULL JOIN`` keeps unmatched rows from both sides, so
        restricting either side to the other's keys would delete rows
        the query asked for."""
        calls, _result = drive_chain(
            build_call_helper(
                "SELECT i.id, c.title FROM items AS i "
                "FULL JOIN cats AS c ON i.cat_id = c.id"
            ),
            [ITEMS, CATS],
        )
        assert "filter" not in calls[1].kwargs


class TestCombination:
    def test_inner_join_matches_rows(self, build_call_helper, drive_chain):
        _calls, (rows, desc, rowcount, _l) = drive_chain(
            build_call_helper(SQL), [ITEMS, CATS]
        )
        assert rows == [(1, "a", "books"), (2, "b", "films")]
        assert [column[0] for column in desc] == ["id", "name", "title"]
        assert rowcount == 2

    def test_left_join_keeps_unmatched_rows(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT i.id, c.title FROM items AS i "
                "LEFT JOIN cats AS c ON i.cat_id = c.id"
            ),
            [ITEMS, [{"id": 10, "title": "books"}]],
        )
        assert rows == [(1, "books"), (2, None)]

    def test_a_null_key_never_matches(self, build_call_helper, drive_chain):
        """SQL's equi-join does not match NULL to NULL, and neither does
        this -- a row whose foreign key is null is unmatched, not paired
        with every other null."""
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT i.id, c.title FROM items AS i "
                "LEFT JOIN cats AS c ON i.cat_id = c.id"
            ),
            [
                [{"id": 1, "cat_id": None}],
                [{"id": None, "title": "orphan"}],
            ],
        )
        assert rows == [(1, None)]

    def test_cross_join_pairs_everything(self, build_call_helper, drive_chain):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT i.id, c.title FROM items AS i CROSS JOIN cats AS c"
            ),
            [[{"id": 1}, {"id": 2}], CATS],
        )
        assert rows == [
            (1, "books"),
            (1, "films"),
            (2, "books"),
            (2, "films"),
        ]

    def test_a_cross_collection_predicate_is_applied_after_the_join(
        self, build_call_helper, drive_chain
    ):
        """``i.price > c.id`` names two collections, so Milvus cannot
        filter on it -- it stays behind and runs over the joined
        frame."""
        calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT i.id FROM items AS i JOIN cats AS c "
                "ON i.cat_id = c.id WHERE i.price > c.id"
            ),
            [ITEMS, CATS],
        )
        assert "filter" not in calls[0].kwargs
        assert rows == [(1,), (2,)]

    def test_order_by_can_name_an_unprojected_column(
        self, build_call_helper, drive_chain
    ):
        """``ORDER BY`` runs before the projection, so it may sort on a
        column the SELECT list never returns."""
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT c.title FROM items AS i "
                "JOIN cats AS c ON i.cat_id = c.id ORDER BY i.price DESC"
            ),
            [ITEMS, CATS],
        )
        assert rows == [("films",), ("books",)]


class TestVectorSearchDrivesTheJoin:
    SQL = (
        "SELECT i.id, c.title FROM items AS i "
        "JOIN cats AS c ON i.cat_id = c.id "
        "ORDER BY i.embedding <=> :q LIMIT 2"
    )

    def test_the_search_runs_first_and_bounds_the_chain(
        self, build_call_helper
    ):
        call = build_call_helper(self.SQL, {"q": [0.1, 0.2]})
        assert call.method == "search"
        assert call.kwargs["collection_name"] == "items"
        assert call.kwargs["anns_field"] == "embedding"
        assert call.kwargs["data"] == [[0.1, 0.2]]
        assert call.kwargs["limit"] == 2
        assert call.kwargs["search_params"]["metric_type"] == "COSINE"

    def test_hits_are_joined_to_the_second_collection(
        self, build_call_helper, drive_chain
    ):
        hits = [
            [
                {"id": 1, "distance": 0.1, "entity": {"cat_id": 10}},
                {"id": 2, "distance": 0.2, "entity": {"cat_id": 11}},
            ]
        ]
        calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(self.SQL, {"q": [0.1, 0.2]}), [hits, CATS]
        )
        assert calls[1].kwargs["filter"] == "id in [10, 11]"
        assert rows == [(1, "books"), (2, "films")]


class TestOuterJoinPredicates:
    """A predicate on the side an outer join can null-extend has to run
    *after* the join. Pushing it into that collection's own filter
    answers a different question -- and used to: every item came back
    from the anti-join below, because Milvus was asked for categories
    whose title is null (none) and the join then null-extended
    everything."""

    ANTI_JOIN = (
        "SELECT i.id FROM items AS i "
        "LEFT JOIN cats AS c ON i.cat_id = c.id WHERE c.title IS NULL"
    )

    def test_a_predicate_on_the_null_side_is_not_pushed_down(
        self, build_call_helper, drive_chain
    ):
        calls, _result = drive_chain(
            build_call_helper(self.ANTI_JOIN), [ITEMS, CATS]
        )
        # The join-key restriction is still there and still safe -- it
        # only drops rows that could not have matched. What must not be
        # there is the `title is null` predicate itself.
        assert calls[1].kwargs["filter"] == "id in [10, 11]"

    def test_the_anti_join_returns_only_unmatched_rows(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(self.ANTI_JOIN),
            [ITEMS, [{"id": 10, "title": "books"}]],
        )
        assert rows == [(2,)]

    def test_a_predicate_on_the_preserved_side_is_still_pushed_down(
        self, build_call_helper
    ):
        """``WHERE`` filters the left side of a LEFT JOIN whether it runs
        before or after the join, so that one stays a Milvus filter."""
        call = build_call_helper(
            "SELECT i.id, c.title FROM items AS i "
            "LEFT JOIN cats AS c ON i.cat_id = c.id WHERE i.price > 20"
        )
        assert call.kwargs["collection_name"] == "items"
        assert call.kwargs["filter"] == "price > 20"

    def test_an_inner_join_pushes_both_sides(
        self, build_call_helper, drive_chain
    ):
        """Nothing is null-extended by an inner join, so a predicate on
        either side is safe to hand to Milvus."""
        calls, _result = drive_chain(
            build_call_helper(
                "SELECT i.id FROM items AS i JOIN cats AS c "
                "ON i.cat_id = c.id WHERE c.title = 'books'"
            ),
            [CATS, ITEMS],
        )
        assert calls[0].kwargs["filter"] == 'title == "books"'


class TestUsing:
    """``USING (k)`` is ``ON left.k = right.k`` with the left side left
    implicit -- resolvable here because exactly one source precedes the
    join."""

    SQL = (
        "SELECT i.id, c.title FROM items AS i JOIN cats AS c USING (cat_id)"
    )

    def test_the_key_is_fetched_from_both_sides(
        self, build_call_helper, drive_chain
    ):
        """Neither side mentions ``cat_id`` in any expression, so the
        planned join keys are what put it in ``output_fields``."""
        calls, _result = drive_chain(
            build_call_helper(self.SQL),
            [
                [{"id": 1, "cat_id": 10}],
                [{"cat_id": 10, "title": "books"}],
            ],
        )
        assert "cat_id" in calls[0].kwargs["output_fields"]
        assert "cat_id" in calls[1].kwargs["output_fields"]

    def test_it_joins_on_that_key(self, build_call_helper, drive_chain):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(self.SQL),
            [
                [{"id": 1, "cat_id": 10}, {"id": 2, "cat_id": 11}],
                [{"cat_id": 10, "title": "books"}],
            ],
        )
        assert rows == [(1, "books")]


class TestCommonTableExpressions:
    """A CTE is a derived table that happens to be named up front, so it
    plans exactly like a subquery in ``FROM``."""

    SQL = (
        "WITH hot AS (SELECT id, cat_id FROM items WHERE price > 40) "
        "SELECT h.id, c.title FROM hot AS h "
        "JOIN cats AS c ON h.cat_id = c.id"
    )

    def test_the_cte_filter_is_pushed_to_milvus(self, build_call_helper):
        call = build_call_helper(self.SQL)
        assert call.kwargs["collection_name"] == "items"
        assert call.kwargs["filter"] == "price > 40"

    def test_the_outer_query_joins_the_cte_result(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(self.SQL),
            [[{"id": 1, "cat_id": 10}], [{"id": 10, "title": "books"}]],
        )
        assert rows == [(1, "books")]

    def test_referencing_one_twice_reads_it_once(
        self, build_call_helper, drive_chain
    ):
        """The CTE's collection is read once; both references evaluate
        against the same fetched frame."""
        calls, _result = drive_chain(
            build_call_helper(
                "WITH hot AS (SELECT id, cat_id FROM items WHERE price > 40) "
                "SELECT a.id FROM hot AS a JOIN hot AS b ON a.cat_id = b.cat_id"
            ),
            [[{"id": 1, "cat_id": 10}]],
        )
        assert len(calls) == 1

    def test_a_cte_used_without_a_join_still_reaches_the_engine(
        self, build_call_helper
    ):
        """Nothing else about this statement needs the engine, so the
        `WITH` itself has to be what routes it there -- otherwise the
        single-call path asks Milvus for a collection named after the
        CTE."""
        call = build_call_helper(
            "WITH hot AS (SELECT id FROM items WHERE price > 40) "
            "SELECT h.id FROM hot AS h"
        )
        assert call.kwargs["collection_name"] == "items"
