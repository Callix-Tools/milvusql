"""Error-path coverage for the full-text surface."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_a_non_bm25_generator_is_rejected_by_name(build_call_helper):
    with pytest.raises(milvusql.NotSupportedError, match="BM25"):
        build_call_helper(
            "CREATE TABLE d (id BIGINT PRIMARY KEY, c TEXT, "
            "cs SPARSEVEC AS (LOWER(c)), v VECTOR(4))"
        )


def test_bm25_generating_anything_but_a_sparsevec_is_rejected(
    build_call_helper,
):
    with pytest.raises(milvusql.NotSupportedError, match="SPARSEVEC"):
        build_call_helper(
            "CREATE TABLE d (id BIGINT PRIMARY KEY, c TEXT, "
            "cs VECTOR(8) AS (BM25(c)))"
        )
