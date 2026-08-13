"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on
``CREATE TABLE``: parsed MilvusQL AST -> the ``Call`` (method, kwargs,
postprocess) it dispatches to."""

from __future__ import annotations

import pytest
from pymilvus import DataType

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_maps_scalar_and_vector_columns(build_call_helper):
    call = build_call_helper(
        "CREATE TABLE items ("
        "id BIGINT PRIMARY KEY AUTO_INCREMENT, "
        "embedding VECTOR(8), "
        "category VARCHAR(64)"
        ")"
    )
    assert call.method == "create_collection"
    assert call.kwargs["collection_name"] == "items"
    fields = {f.name: f for f in call.kwargs["schema"].fields}
    assert fields["id"].dtype == DataType.INT64
    assert fields["id"].is_primary is True
    assert fields["id"].auto_id is True
    assert fields["embedding"].dtype == DataType.FLOAT_VECTOR
    assert fields["embedding"].params["dim"] == 8
    assert fields["category"].dtype == DataType.VARCHAR
    assert fields["category"].params["max_length"] == 64


def test_sparsevec_column_maps_to_sparse_float_vector(build_call_helper):
    """``SPARSEVEC`` has no dedicated sqlglot ``DataType.Type`` (unlike
    ``VECTOR``) -- it parses as a generic user-defined type carrying its
    own spelling as free text, matched case-insensitively in
    ``_map_datatype``."""
    call = build_call_helper(
        "CREATE TABLE docs (id BIGINT PRIMARY KEY, sparse SPARSEVEC)"
    )
    fields = {f.name: f for f in call.kwargs["schema"].fields}
    assert fields["sparse"].dtype == DataType.SPARSE_FLOAT_VECTOR


def test_sparsevec_column_type_name_is_case_insensitive(build_call_helper):
    call = build_call_helper(
        "CREATE TABLE docs (id BIGINT PRIMARY KEY, sparse sparsevec)"
    )
    fields = {f.name: f for f in call.kwargs["schema"].fields}
    assert fields["sparse"].dtype == DataType.SPARSE_FLOAT_VECTOR


def test_standalone_primary_key_constraint_is_recognized(build_call_helper):
    """SQLAlchemy's own ``DDLCompiler`` emits a standalone
    ``PRIMARY KEY (col)`` rather than an inline keyword by default
    (confirmed directly, noted in ``milvusql_sqlalchemy.compiler``'s
    own docstring) -- this is the shape that produces."""
    call = build_call_helper(
        "CREATE TABLE items (id BIGINT, category VARCHAR(64), "
        "embedding VECTOR(8), PRIMARY KEY (id))"
    )
    fields = {f.name: f for f in call.kwargs["schema"].fields}
    assert fields["id"].is_primary is True


def test_with_properties_map_shards_and_pass_through_consistency_level(
    build_call_helper,
):
    call = build_call_helper(
        "CREATE TABLE items (id BIGINT PRIMARY KEY, category VARCHAR(8), "
        "embedding VECTOR(8)) "
        "WITH (shards=2, consistency_level='Bounded')"
    )
    assert call.kwargs["num_shards"] == 2
    assert call.kwargs["consistency_level"] == "Bounded"


def test_partition_key_property_is_consumed_building_the_schema(
    build_call_helper,
):
    call = build_call_helper(
        "CREATE TABLE items (id BIGINT PRIMARY KEY, category VARCHAR(8), "
        "embedding VECTOR(8)) "
        "WITH (partition_key='category')"
    )
    assert "partition_key" not in call.kwargs
    fields = {f.name: f for f in call.kwargs["schema"].fields}
    assert fields["category"].is_partition_key is True


def test_ddl_postprocess_returns_no_rows(build_call_helper):
    call = build_call_helper(
        "CREATE TABLE items (id BIGINT PRIMARY KEY, embedding VECTOR(8))"
    )
    assert call.postprocess(object()) == ([], None, -1, None)
