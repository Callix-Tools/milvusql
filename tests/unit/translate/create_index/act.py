"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on
``CREATE INDEX``."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestCreateIndex:
    def test_maps_using_and_with_knobs_to_index_params(
        self, build_call_helper
    ):
        call = build_call_helper(
            "CREATE INDEX idx_emb ON items (embedding) USING HNSW "
            "WITH (metric_type='COSINE', M=16, ef_construction=200)"
        )
        assert call.method == "create_index"
        assert call.kwargs["collection_name"] == "items"
        assert [p.to_dict() for p in call.kwargs["index_params"]] == [
            {
                "field_name": "embedding",
                "index_type": "HNSW",
                "index_name": "idx_emb",
                "metric_type": "COSINE",
                "M": 16,
                "ef_construction": 200,
            }
        ]
