"""Error-path coverage for ``translate.ast_to_pymilvus.build_call`` on
a plain ``SELECT`` -- shapes Milvus genuinely cannot execute, which
must raise a clean ``NotSupportedError`` instead of silently narrowing
the query to something the caller never wrote."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestJoinIsRejected:
    def test_join_raises_not_supported_error(self, build_call_helper):
        """Milvus has no cross-collection JOIN. Left unchecked, the
        table name was read straight off ``FROM``'s own ``this`` and
        the ``JOIN`` clause was never consulted -- the query silently
        ran against the first table only, as if ``JOIN ...`` had never
        been written."""
        with pytest.raises(milvusql.NotSupportedError, match="JOIN"):
            build_call_helper(
                "SELECT id FROM items JOIN other ON items.id = other.item_id"
            )


class TestSubqueryFromIsRejected:
    def test_subquery_with_its_own_join_raises_not_supported_error(
        self, build_call_helper
    ):
        """Milvus has no subquery-as-source. A subquery whose own
        ``FROM`` isn't a single plain table (here: it has a ``JOIN``
        of its own) is not the trivial pass-through shape
        ``_flatten_trivial_subquery`` unwraps, so this is still
        genuinely unsupported -- left unchecked, reading ``.name`` off
        a ``Subquery`` node would not fail cleanly."""
        with pytest.raises(milvusql.NotSupportedError, match="subquery"):
            build_call_helper(
                "SELECT id FROM (SELECT id FROM items "
                "JOIN other ON items.id = other.item_id) AS sub"
            )

    def test_subquery_with_its_own_limit_raises_not_supported_error(
        self, build_call_helper
    ):
        """A ``LIMIT`` inside the subquery genuinely changes which
        rows reach the outer query -- flattening it away would be
        silently wrong, not just permissive."""
        with pytest.raises(milvusql.NotSupportedError, match="subquery"):
            build_call_helper(
                "SELECT id FROM (SELECT id FROM items LIMIT 1) AS sub"
            )

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
