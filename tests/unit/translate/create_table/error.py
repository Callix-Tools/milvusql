"""Error-path coverage for ``CREATE TABLE`` dispatch."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_unsupported_column_type_raises_not_supported_error(build_call_helper):
    """``DATETIME`` has no Milvus field type at all -- the honest
    rejection an unmapped scalar spelling has always gotten (``TEXT``
    used to be this test's example, but it maps to an analyzer-enabled
    VARCHAR now)."""
    with pytest.raises(milvusql.NotSupportedError, match="DATETIME"):
        build_call_helper(
            "CREATE TABLE items (id BIGINT PRIMARY KEY, "
            "payload DATETIME, v VECTOR(4))"
        )


def test_unrecognized_user_defined_type_still_raises_not_supported_error(
    build_call_helper,
):
    """``SPARSEVEC`` is one specific, recognized spelling of the generic
    ``USERDEFINED`` shape ``_map_datatype`` now matches -- any other
    unrecognized user-defined type name must still raise, not silently
    fall through to some other Milvus type."""
    with pytest.raises(milvusql.NotSupportedError, match="NOTAREALTYPE"):
        build_call_helper("CREATE TABLE items (payload NOTAREALTYPE)")


def test_no_vector_column_raises_not_supported_error(build_call_helper):
    """Milvus refuses ``create_collection()`` outright for a schema
    with zero vector fields -- confirmed directly against a real,
    non-Lite server (Milvus Lite silently tolerated it). Raised here,
    client-side, before any RPC goes out, with a clear message instead
    of Milvus's own opaque low-level rejection -- notably, this means
    a tool that creates a table with no vector column of its own (e.g.
    Alembic's ``alembic_version`` bookkeeping table) can never be made
    to work against this backend; see
    ``milvusql_sqlalchemy``'s own ``alembic_migration`` integration
    coverage for that specific, deliberate consequence."""
    with pytest.raises(milvusql.NotSupportedError, match="VECTOR"):
        build_call_helper(
            "CREATE TABLE items (id BIGINT PRIMARY KEY, category VARCHAR(64))"
        )
