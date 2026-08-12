"""Error-path coverage for ``CREATE TABLE`` dispatch."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_unsupported_column_type_raises_not_supported_error(build_call_helper):
    with pytest.raises(milvusql.NotSupportedError, match="TEXT"):
        build_call_helper("CREATE TABLE items (payload TEXT)")
