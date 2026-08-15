"""Unit coverage for correlated ``[NOT] EXISTS`` -- decorrelated into a
semi/anti join, the shape SQLAlchemy's ``.any()``/``.has()`` and
Django's ``Exists(... OuterRef(...))`` compile to. The exact SQL texts
here are the ones those ORMs emit (confirmed against their compilers),
so this is the ORM contract tested without the ORM in the loop."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]

ITEMS = [
    {"id": 1, "name": "a"},
    {"id": 2, "name": "b"},
    {"id": 3, "name": "c"},
]
CATS = [{"item_id": 1}, {"item_id": 3}]

#: SQLAlchemy `.any()` / `.has()`: EXISTS with an equality correlation.
SQLALCHEMY_ANY = (
    "SELECT items.id, items.name FROM items WHERE EXISTS "
    "(SELECT 1 FROM cats WHERE items.id = cats.item_id "
    "AND cats.active = true)"
)

#: Django `Exists(OuterRef(...))`: aliased subquery, `SELECT 1 AS "a"`,
#: quoted everything, and a LIMIT 1 the compiler always adds.
DJANGO_EXISTS = (
    'SELECT "items"."id" FROM "items" WHERE NOT EXISTS '
    '(SELECT 1 AS "a" FROM "cats" U0 '
    'WHERE U0."item_id" = "items"."id" LIMIT 1)'
)


class TestSemiJoin:
    def test_exists_keeps_only_outer_rows_with_a_match(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, rowcount, _l) = drive_chain(
            build_call_helper(SQLALCHEMY_ANY), [CATS, ITEMS]
        )
        assert rows == [(1, "a"), (3, "c")]
        assert rowcount == 2

    def test_the_inner_only_predicate_is_pushed_to_milvus(
        self, build_call_helper, drive_chain
    ):
        """``cats.active = true`` names only the subquery's collection,
        so it travels to Milvus as that scan's own filter; only the
        correlation itself is evaluated client-side."""
        calls, _result = drive_chain(
            build_call_helper(SQLALCHEMY_ANY), [CATS, ITEMS]
        )
        cats_scan = next(
            c for c in calls if c.kwargs.get("collection_name") == "cats"
        )
        assert cats_scan.kwargs["filter"] == "active == true"
        assert cats_scan.kwargs["output_fields"] == ["item_id"]

    def test_not_exists_is_the_anti_join(self, build_call_helper, drive_chain):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(DJANGO_EXISTS), [CATS, ITEMS]
        )
        assert rows == [(2,)]

    def test_djangos_inner_limit_one_does_not_starve_the_join(
        self, build_call_helper, drive_chain
    ):
        """Django spells the subquery ``SELECT 1 ... LIMIT 1``. Applied
        to the *decorrelated* set that LIMIT would keep one key row for
        the whole query; per outer row it was always inert, so it is
        dropped -- both matching outer rows must survive."""
        sql = DJANGO_EXISTS.replace("NOT EXISTS", "EXISTS")
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(sql), [CATS, ITEMS]
        )
        assert rows == [(1,), (3,)]

    def test_a_null_outer_key_never_matches_exists(
        self, build_call_helper, drive_chain
    ):
        """SQL's own semantics: ``NULL = anything`` is unknown, so
        ``EXISTS`` drops the row and ``NOT EXISTS`` keeps it."""
        items = [*ITEMS, {"id": None, "name": "ghost"}]
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(SQLALCHEMY_ANY), [CATS, items]
        )
        assert rows == [(1, "a"), (3, "c")]
        _calls, (kept, _d, _rc, _l) = drive_chain(
            build_call_helper(DJANGO_EXISTS), [CATS, items]
        )
        assert kept == [(2,), (None,)]

    def test_an_uncorrelated_exists_still_evaluates_whole_frame(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT items.id FROM items WHERE EXISTS "
                "(SELECT id FROM cats WHERE cats.active = true)"
            ),
            [CATS, ITEMS],
        )
        assert rows == [(1,), (2,), (3,)]
