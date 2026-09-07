"""Error-path coverage for ``translate.ast_to_pymilvus.build_call`` on
a plain ``SELECT`` -- shapes Milvus genuinely cannot execute, which
must raise a clean ``NotSupportedError`` instead of silently narrowing
the query to something the caller never wrote."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestUnsupportedSources:
    def test_no_from_clause_at_all_raises_not_supported_error(
        self, build_call_helper
    ):
        """A bare ``SELECT EXISTS(...)`` (SQLAlchemy's ``Query.exists()``)
        has no ``FROM`` at all -- every collection-reading builder
        needs one. Left unchecked, ``ast.args["from_"]`` raised a raw
        ``KeyError`` instead of a DBAPI exception."""
        with pytest.raises(milvusql.NotSupportedError, match="FROM"):
            build_call_helper(
                "SELECT EXISTS(SELECT 1 FROM items WHERE id = 1)"
            )


class TestIsComparisonIsRejectedForAnythingButNull:
    def test_is_true_raises_not_supported_error(self, build_call_helper):
        """Milvus's filter DSL only has ``is null``/``is not null`` --
        no other ``IS <predicate>`` form (e.g. ``IS TRUE``) to
        transpile to."""
        with pytest.raises(milvusql.NotSupportedError, match="IS"):
            build_call_helper(
                "SELECT id FROM items WHERE (category = :cat) IS TRUE",
                {"cat": "book"},
            )


class TestComputedProjectionsAreRejected:
    """``output_fields`` is a list of stored field names, not an
    expression language, so a computed SELECT-list element has no
    faithful translation. Every one of these used to compile: the
    element's ``.name`` is ``''`` for a non-column node, so Milvus was
    asked for a field named ``''`` -- dropped from the response without
    an error, which surfaced as a column of ``None``s under the
    expression's own label."""

    def test_arithmetic_in_the_select_list_raises(self, build_call_helper):
        with pytest.raises(milvusql.NotSupportedError, match="price \\* 2"):
            build_call_helper("SELECT id, price * 2 FROM items LIMIT 5")

    def test_a_function_call_in_the_select_list_raises(
        self, build_call_helper
    ):
        with pytest.raises(milvusql.NotSupportedError, match="UPPER"):
            build_call_helper(
                "SELECT id, UPPER(category) AS c FROM items LIMIT 5"
            )

    def test_the_message_says_what_to_do_instead(self, build_call_helper):
        """Rejecting is only half of it -- the message has to name the
        way through, the same way the filter-path rejections do."""
        with pytest.raises(
            milvusql.NotSupportedError,
            match="computes nothing in a projection",
        ):
            build_call_helper("SELECT id, price + 1 AS p FROM items LIMIT 5")

    def test_columns_star_and_literals_still_compile(self, build_call_helper):
        """The guard must not narrow what already worked: a bare
        column, an aliased column, ``*`` and a constant all name
        something Milvus can return without computing anything."""
        call = build_call_helper(
            "SELECT id, category AS c, 1 AS one FROM items LIMIT 5"
        )
        assert call.kwargs["output_fields"] == ["id", "category", "1"]
        assert build_call_helper("SELECT * FROM items LIMIT 5").kwargs[
            "output_fields"
        ] == ["*"]
