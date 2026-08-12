"""Error-path coverage for ``CREATE TABLE`` dispatch."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_unsupported_column_type_raises_not_supported_error(build_call_helper):
    with pytest.raises(milvusql.NotSupportedError, match="TEXT"):
        build_call_helper("CREATE TABLE items (payload TEXT)")


def test_unrecognized_user_defined_type_still_raises_not_supported_error(
    build_call_helper,
):
    """``SPARSEVEC`` is one specific, recognized spelling of the generic
    ``USERDEFINED`` shape ``_map_datatype`` now matches -- any other
    unrecognized user-defined type name must still raise, not silently
    fall through to some other Milvus type."""
    with pytest.raises(milvusql.NotSupportedError, match="NOTAREALTYPE"):
        build_call_helper("CREATE TABLE items (payload NOTAREALTYPE)")
