"""Error-path coverage for subqueries: the one class of them this
engine cannot answer, and why the message says so by name."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestCorrelatedSubqueries:
    """A correlated subquery's result depends on the outer row being
    evaluated. Answering it means either one Milvus read per outer row
    or a rewrite into a semi-join -- neither exists here, and both ORMs
    emit this shape, so the error names the construct instead of
    surfacing a confusing "unknown table qualifier"."""

    def test_a_correlated_exists_is_rejected(self, build_call_helper):
        """What SQLAlchemy's ``.has()``/``.any()`` and Django's
        ``Exists(... OuterRef(...))`` both compile to."""
        with pytest.raises(milvusql.NotSupportedError, match="correlated"):
            build_call_helper(
                "SELECT i.id FROM items AS i WHERE EXISTS "
                "(SELECT c.id FROM cats AS c WHERE c.id = i.cat_id)"
            )

    def test_a_correlated_scalar_subquery_is_rejected(
        self, build_call_helper
    ):
        """Django's ``annotate(Subquery(...))`` puts one in the SELECT
        list, where the single-call path used to ask Milvus for an
        output field named after it and hand back a column of
        ``None``."""
        with pytest.raises(milvusql.NotSupportedError, match="correlated"):
            build_call_helper(
                "SELECT i.id, (SELECT c.title FROM cats AS c "
                "WHERE c.id = i.cat_id) AS t FROM items AS i"
            )

    def test_the_message_names_the_enclosing_table(self, build_call_helper):
        with pytest.raises(milvusql.NotSupportedError, match="'i'"):
            build_call_helper(
                "SELECT i.id FROM items AS i WHERE EXISTS "
                "(SELECT c.id FROM cats AS c WHERE c.id = i.cat_id)"
            )

    def test_an_uncorrelated_subquery_in_the_select_list_works(
        self, build_call_helper, drive_chain
    ):
        """The uncorrelated version is a value like any other: read it
        once, then use it for every row."""
        calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT i.id, (SELECT MAX(c.id) FROM cats AS c) AS newest "
                "FROM items AS i"
            ),
            [[{"id": 10}, {"id": 11}], [{"id": 1}, {"id": 2}]],
        )
        assert [call.kwargs["collection_name"] for call in calls] == [
            "cats",
            "items",
        ]
        assert rows == [(1, 11), (2, 11)]
