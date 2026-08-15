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
