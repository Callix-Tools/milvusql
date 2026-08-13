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
    def test_subquery_in_from_raises_not_supported_error(
        self, build_call_helper
    ):
        """Milvus has no subquery-as-source. Left unchecked, reading
        ``.name`` off a ``Subquery`` node does not fail cleanly."""
        with pytest.raises(milvusql.NotSupportedError, match="subquery"):
            build_call_helper("SELECT id FROM (SELECT id FROM items) AS sub")
