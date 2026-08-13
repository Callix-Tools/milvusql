"""Error-path coverage for ``translate.ast_to_pymilvus.build_call`` on
``GROUP BY`` -- a shape Milvus cannot execute (no server-side grouping,
nothing downstream reduces per-group), so it must raise a clean
``NotSupportedError`` instead of silently running ungrouped."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestGroupByIsRejected:
    def test_group_by_raises_not_supported_error(self, build_call_helper):
        """``_is_aggregate_select`` already returns ``False`` whenever
        ``GROUP BY`` is present, so this used to fall through to the
        plain-select path and execute as an ungrouped ``query()`` --
        one row per matching entity instead of one row per group, with
        no error at all."""
        with pytest.raises(milvusql.NotSupportedError, match="GROUP BY"):
            build_call_helper(
                "SELECT category, COUNT(*) FROM items GROUP BY category"
            )

    def test_group_by_is_rejected_even_with_a_bare_column_select(
        self, build_call_helper
    ):
        with pytest.raises(milvusql.NotSupportedError, match="GROUP BY"):
            build_call_helper("SELECT category FROM items GROUP BY category")
