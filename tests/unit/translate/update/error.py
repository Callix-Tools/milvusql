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
    def test_an_update_at_the_row_ceiling_pages_then_upserts_everything(
        self, build_call_helper
    ):
        """The read that feeds the upsert pages past the per-call
        ceiling (it used to raise there), so a broad UPDATE reads every
        matching row before anything is written -- and the write itself
        goes out in bounded chunks."""
        call = build_call_helper("UPDATE items SET cid = 1 WHERE id > 0")
        at_ceiling = [{"id": i, "cid": 9} for i in range(DEFAULT_QUERY_LIMIT)]
        describe_call = call.then(at_ceiling)
        assert describe_call.method == "describe_collection"
        page = describe_call.then(
            {"fields": [{"name": "id", "is_primary": True}]}
        )
        assert page.method == "query"
        remainder = [{"id": DEFAULT_QUERY_LIMIT, "cid": 9}]
        upsert = page.then(at_ceiling).then(remainder)
        assert upsert.method == "upsert"
        # 16385 merged rows go out in 1000-row chunks.
        assert len(upsert.kwargs["data"]) == 1000
        assert upsert.kwargs["data"][0] == {"id": 0, "cid": 1}

    def test_a_normal_update_still_upserts(self, build_call_helper):
        call = build_call_helper("UPDATE items SET cid = 1 WHERE id > 0")
        follow_up = call.then([{"id": 1, "cid": 9}])
        assert follow_up is not None
        assert follow_up.method == "upsert"
        assert follow_up.kwargs["data"] == [{"id": 1, "cid": 1}]
