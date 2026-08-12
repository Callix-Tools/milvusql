"""Unit coverage for ``VECTOR``/``SPARSEVEC``: column-spec rendering,
the identity bind/result processors (D1's DBAPI takes real Python
values, never a serialized-to-text vector), and each comparator's
operator spelling -- observed the only way a caller ever sees it,
through a compiled ``ORDER BY`` clause, entirely offline."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.dialect import MilvusDialect
from milvusql_sqlalchemy.types import SPARSEVEC, VECTOR
from sqlalchemy import BigInteger, Column, MetaData, Table, select

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]


class TestGetColSpec:
    def test_vector_with_dim_renders_the_dimension(self):
        assert VECTOR(8).get_col_spec() == "VECTOR(8)"

    def test_vector_without_dim_renders_bare_keyword(self):
        assert VECTOR().get_col_spec() == "VECTOR"

    def test_sparsevec_renders_bare_keyword(self):
        assert SPARSEVEC().get_col_spec() == "SPARSEVEC"


class TestBindAndResultProcessors:
    """Identity round-trip: D1's DBAPI takes real Python values, so
    these only ever copy a list through, never serialize it."""

    def test_bind_processor_copies_the_list_through_unchanged(self):
        process = VECTOR(8).bind_processor(dialect=MilvusDialect())
        assert process([0.1, 0.2]) == [0.1, 0.2]

    def test_bind_processor_passes_none_through(self):
        process = VECTOR(8).bind_processor(dialect=MilvusDialect())
        assert process(None) is None

    def test_result_processor_copies_the_list_through_unchanged(self):
        process = VECTOR(8).result_processor(
            dialect=MilvusDialect(), coltype=None
        )
        assert process([0.1, 0.2]) == [0.1, 0.2]

    def test_result_processor_passes_none_through(self):
        process = VECTOR(8).result_processor(
            dialect=MilvusDialect(), coltype=None
        )
        assert process(None) is None


def _compiled_order_by(column_op) -> str:
    metadata = MetaData()
    items = Table(
        "items",
        metadata,
        Column("id", BigInteger, primary_key=True),
        Column("embedding", VECTOR(8)),
        Column("sparse", SPARSEVEC()),
    )
    stmt = select(items.c.id).order_by(column_op(items)).limit(3)
    return str(stmt.compile(dialect=MilvusDialect()))


class TestVectorComparator:
    def test_l2_distance_renders_the_diamond_arrow_operator(self):
        sql = _compiled_order_by(
            lambda t: t.c.embedding.l2_distance([0.1] * 8)
        )
        assert "items.embedding <-> :embedding_1" in sql

    def test_cosine_distance_renders_the_double_diamond_operator(self):
        sql = _compiled_order_by(
            lambda t: t.c.embedding.cosine_distance([0.1] * 8)
        )
        assert "items.embedding <=> :embedding_1" in sql

    def test_max_inner_product_renders_the_hash_diamond_operator(self):
        sql = _compiled_order_by(
            lambda t: t.c.embedding.max_inner_product([0.1] * 8)
        )
        assert "items.embedding <#> :embedding_1" in sql

    def test_l1_distance_renders_the_plus_diamond_operator(self):
        sql = _compiled_order_by(
            lambda t: t.c.embedding.l1_distance([0.1] * 8)
        )
        assert "items.embedding <+> :embedding_1" in sql


class TestSparsevecComparator:
    def test_max_inner_product_renders_the_hash_diamond_operator(self):
        sql = _compiled_order_by(
            lambda t: t.c.sparse.max_inner_product({0: 0.5})
        )
        assert "items.sparse <#> :sparse_1" in sql
