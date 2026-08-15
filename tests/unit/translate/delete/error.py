"""Error-path coverage for ``DELETE`` dispatch."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_in_subquery_is_rejected_not_rendered_false(build_call_helper):
    """`IN (SELECT ...)` keeps its subquery in `query`, not
    `expressions` -- the empty-IN branch used to render it as the
    constant `false`, which made `NOT IN (SELECT ...)` delete every
    row (confirmed by direct execution). Both spellings now get a
    named rejection before anything reaches the server."""
    with pytest.raises(milvusql.NotSupportedError, match="IN \\(SELECT"):
        build_call_helper("DELETE FROM items WHERE id IN (SELECT id FROM s)")
    with pytest.raises(milvusql.NotSupportedError, match="IN \\(SELECT"):
        build_call_helper(
            "DELETE FROM items WHERE id NOT IN (SELECT id FROM keep)"
        )
