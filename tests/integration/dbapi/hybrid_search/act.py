"""Integration coverage for ``HYBRID SEARCH`` against a real (embedded)
Milvus Lite instance -- two indexed vector fields, combined and
reranked, not just the ``build_call`` dispatch shape unit tests cover."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]

CREATE_DOCS = (
    "CREATE TABLE docs ("
    "id BIGINT PRIMARY KEY AUTO_INCREMENT, "
    "dense VECTOR(4), "
    "sparse VECTOR(4), "
    "category VARCHAR(64)"
    ") WITH (shards=1)"
)


@pytest.fixture
def loaded_docs(conn, cur):
    cur.execute(CREATE_DOCS)
    cur.execute(
        "CREATE INDEX idx_dense ON docs (dense) USING HNSW "
        "WITH (metric_type='COSINE', M=16, ef_construction=200)"
    )
    cur.execute(
        "CREATE INDEX idx_sparse ON docs (sparse) USING HNSW "
        "WITH (metric_type='COSINE', M=16, ef_construction=200)"
    )
    cur.execute("LOAD TABLE docs")
    cur.execute(
        "INSERT INTO docs (dense, sparse, category) VALUES (:d, :s, :c)",
        {"d": [0.1, 0.1, 0.1, 0.1], "s": [0.9, 0.9, 0.9, 0.9], "c": "a"},
    )
    cur.execute(
        "INSERT INTO docs (dense, sparse, category) VALUES (:d, :s, :c)",
        {"d": [0.9, 0.9, 0.9, 0.9], "s": [0.1, 0.1, 0.1, 0.1], "c": "b"},
    )
    return conn


def test_hybrid_search_returns_a_rank_fused_result(loaded_docs, cur):
    cur.execute(
        "SELECT id, category FROM docs HYBRID SEARCH "
        "(dense <=> :d WEIGHT 0.5, sparse <=> :s WEIGHT 0.5) "
        "RERANK RRF LIMIT 5",
        {"d": [0.1, 0.1, 0.1, 0.1], "s": [0.9, 0.9, 0.9, 0.9]},
    )
    rows = cur.fetchall()
    assert {row[0] for row in rows} == {1, 2}


def test_hybrid_search_applies_the_shared_filter_to_every_arm(
    loaded_docs, cur
):
    cur.execute(
        "SELECT id FROM docs WHERE category = :cat HYBRID SEARCH "
        "(dense <=> :d WEIGHT 0.5, sparse <=> :s WEIGHT 0.5) "
        "RERANK RRF LIMIT 5",
        {
            "cat": "a",
            "d": [0.1, 0.1, 0.1, 0.1],
            "s": [0.9, 0.9, 0.9, 0.9],
        },
    )
    assert cur.fetchall() == [(1,)]
