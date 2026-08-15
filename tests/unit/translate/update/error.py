"""Error-path coverage for ``UPDATE`` dispatch."""

from __future__ import annotations

import pytest

import milvusql
from milvusql.translate._common import DEFAULT_QUERY_LIMIT

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_a_computed_set_value_is_rejected(build_call_helper):
    """``UPDATE ... SET col = col + 1`` (an ``F()`` expression, in
    Django terms) is still one ``SET col = <expr>`` assignment, just
    not one whose right-hand side is a bare bind value or literal --
    the same ``_resolve_value`` every other value in this module
    already goes through, rejected the same way rather than silently
    resolving ``stock`` as an unbound column reference."""
    with pytest.raises(
        milvusql.NotSupportedError, match="unsupported value expression"
    ):
        build_call_helper('UPDATE items SET "stock" = "stock" + 1')


class TestUpdateTruncation:
    def test_an_update_at_the_row_ceiling_raises(self, build_call_helper):
        """The read that feeds the upsert is subject to the same ceiling
        as any other read. Writing back only what fit would leave the
        rest untouched and report a rowcount that looks complete --
        nothing has been written when this raises."""
        call = build_call_helper("UPDATE items SET cid = 1 WHERE id > 0")
        at_ceiling = [{"id": i} for i in range(DEFAULT_QUERY_LIMIT)]
        with pytest.raises(milvusql.NotSupportedError, match="row ceiling"):
            call.then(at_ceiling)

    def test_a_normal_update_still_upserts(self, build_call_helper):
        call = build_call_helper("UPDATE items SET cid = 1 WHERE id > 0")
        follow_up = call.then([{"id": 1, "cid": 9}])
        assert follow_up is not None
        assert follow_up.method == "upsert"
        assert follow_up.kwargs["data"] == [{"id": 1, "cid": 1}]
