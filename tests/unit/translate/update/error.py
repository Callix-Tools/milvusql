"""Error-path coverage for ``UPDATE`` dispatch."""

from __future__ import annotations

import pytest

import milvusql

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
