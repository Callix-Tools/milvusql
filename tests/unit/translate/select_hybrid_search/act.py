"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on
``HYBRID SEARCH`` -- dispatched to ``MilvusClient.hybrid_search`` with
one ``AnnSearchRequest`` per arm and a reranker built from the
``RERANK`` clause."""

from __future__ import annotations

import pytest
from pymilvus import AnnSearchRequest

pytestmark = [pytest.mark.unit, pytest.mark.translate]

SQL = (
    "SELECT id, category FROM items WHERE category = :cat HYBRID SEARCH "
    "(embedding <=> :dv WEIGHT 0.7, sparse <#> :sv WEIGHT 0.3) "
    "RERANK RRF LIMIT 5"
)
PARAMS = {"cat": "book", "dv": [0.1] * 8, "sv": [0.2] * 8}


def test_dispatches_to_hybrid_search(build_call_helper):
    call = build_call_helper(SQL, PARAMS)
    assert call.method == "hybrid_search"
    assert call.kwargs["collection_name"] == "items"
    assert call.kwargs["limit"] == 5
    assert call.kwargs["output_fields"] == ["id", "category"]


def test_builds_one_ann_search_request_per_arm_with_the_shared_filter(
    build_call_helper,
):
    call = build_call_helper(SQL, PARAMS)
    reqs = call.kwargs["reqs"]
    assert len(reqs) == 2
    assert all(isinstance(r, AnnSearchRequest) for r in reqs)
    assert reqs[0].anns_field == "embedding"
    assert reqs[0].param == {"metric_type": "COSINE"}
    assert reqs[0].expr == 'category == "book"'
    assert reqs[1].anns_field == "sparse"
    assert reqs[1].param == {"metric_type": "IP"}


def test_rerank_rrf_is_the_default_ranker(build_call_helper):
    call = build_call_helper(SQL, PARAMS)
    ranker = call.kwargs["ranker"]
    assert ranker._strategy == "rrf"
    assert ranker._k == 60


def test_rerank_rrf_accepts_an_explicit_k(build_call_helper):
    call = build_call_helper(
        "SELECT id FROM items HYBRID SEARCH (embedding <=> :dv) "
        "RERANK RRF(k=30) LIMIT 5",
        {"dv": [0.1] * 8},
    )
    assert call.kwargs["ranker"]._k == 30


def test_rerank_weighted_carries_each_arms_weight_in_arm_order(
    build_call_helper,
):
    call = build_call_helper(
        "SELECT id FROM items HYBRID SEARCH "
        "(embedding <=> :dv WEIGHT 0.7, sparse <#> :sv WEIGHT 0.3) "
        "RERANK WEIGHTED LIMIT 5",
        {"dv": [0.1] * 8, "sv": [0.2] * 8},
    )
    ranker = call.kwargs["ranker"]
    assert ranker._strategy == "weighted"
    assert ranker._weights == [0.7, 0.3]


def test_hybrid_search_postprocess_reads_the_top_level_hit_list(
    build_call_helper,
):
    call = build_call_helper(SQL, PARAMS)
    raw = [[{"entity": {"category": "book"}, "id": 1, "distance": 0.01}]]
    rows, _description, rowcount, _lastrowid = call.postprocess(raw)
    assert rows == [(1, "book")]
    assert rowcount == 1
