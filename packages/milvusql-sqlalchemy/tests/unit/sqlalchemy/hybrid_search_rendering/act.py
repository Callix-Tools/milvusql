"""Unit coverage for ``MilvusSQLCompiler``'s ``hybrid.py`` construct:
``HYBRID SEARCH (...) RERANK ...`` rendering (D3). Entirely offline --
``some_stmt.compile(dialect=MilvusDialect())`` does zero network I/O.

HYBRID SEARCH is not executable end to end yet (the root ``milvusql``
package's ``build_call`` raises ``NotSupportedError`` for it -- see
``tests/test_translate.py::TestSelectVectorSearch::test_hybrid_search_is_not_supported_yet``
at the repo root), so it only gets this compiled-text coverage, never
an integration test against a real engine."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.dialect import MilvusDialect
from milvusql_sqlalchemy.hybrid import hybrid_search, weighted
from milvusql_sqlalchemy.types import SPARSEVEC, VECTOR
from sqlalchemy import BigInteger, Column, MetaData, Table, select

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]


@pytest.fixture
def dialect() -> MilvusDialect:
    return MilvusDialect()


@pytest.fixture
def items_table():
    metadata = MetaData()
    return Table(
        "items",
        metadata,
        Column("id", BigInteger, primary_key=True),
        Column("embedding", VECTOR(8)),
        Column("sparse", SPARSEVEC()),
    )


class TestHybridSearchRendering:
    def test_two_arms_with_rerank_params(self, dialect, items_table):
        stmt = (
            select(items_table.c.id)
            .order_by(
                hybrid_search(
                    weighted(
                        items_table.c.embedding.cosine_distance([0.1] * 8),
                        0.7,
                    ),
                    weighted(
                        items_table.c.sparse.max_inner_product({0: 0.5}),
                        0.3,
                    ),
                    rerank="RRF",
                    k=60,
                )
            )
            .limit(10)
        )
        compiled = stmt.compile(dialect=dialect)
        assert (
            "HYBRID SEARCH (items.embedding <=> :embedding_1 WEIGHT 0.7, "
            "items.sparse <#> :sparse_1 WEIGHT 0.3) RERANK RRF(k=60)"
        ) in str(compiled)
        assert compiled.params == {
            "embedding_1": [0.1] * 8,
            "sparse_1": {0: 0.5},
            "param_1": 10,
        }

    def test_single_arm_without_rerank_params_omits_the_parens(
        self, dialect, items_table
    ):
        stmt = (
            select(items_table.c.id)
            .order_by(
                hybrid_search(
                    weighted(
                        items_table.c.embedding.cosine_distance([0.1] * 8),
                        0.7,
                    ),
                    rerank="RRF",
                )
            )
            .limit(5)
        )
        sql = str(stmt.compile(dialect=dialect))
        assert (
            "HYBRID SEARCH (items.embedding <=> :embedding_1 WEIGHT 0.7) "
            "RERANK RRF"
        ) in sql
        assert "RRF(" not in sql

    def test_renders_without_the_order_by_keyword(self, dialect, items_table):
        """D3: ``HYBRID SEARCH ... RERANK ...`` sits exactly where
        ``ORDER BY`` would, so MilvusQL uses no separate keyword for
        it -- ``order_by_clause``'s override."""
        stmt = select(items_table.c.id).order_by(
            hybrid_search(
                weighted(
                    items_table.c.embedding.cosine_distance([0.1] * 8), 1.0
                ),
                rerank="RRF",
            )
        )
        sql = str(stmt.compile(dialect=dialect))
        assert "ORDER BY" not in sql
        assert "HYBRID SEARCH" in sql
