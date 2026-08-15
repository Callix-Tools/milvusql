"""Unit coverage for the full-text surface: the ``TEXT`` column and its
BM25-generated ``SPARSEVEC`` in DDL, ``BM25_SCORE`` as a search, and
``MATCH ... AGAINST`` as a filter."""

from __future__ import annotations

import pytest
from pymilvus import DataType

pytestmark = [pytest.mark.unit, pytest.mark.translate]

DDL = (
    "CREATE TABLE docs ("
    "id BIGINT PRIMARY KEY AUTO_INCREMENT, "
    "content TEXT, "
    "content_sparse SPARSEVEC GENERATED ALWAYS AS (BM25(content)), "
    "embedding VECTOR(8)"
    ")"
)


class TestDDL:
    def test_text_maps_to_analyzer_enabled_varchar(self, build_call_helper):
        schema = build_call_helper(DDL).kwargs["schema"]
        content = next(f for f in schema.fields if f.name == "content")
        assert content.dtype == DataType.VARCHAR
        assert content.params["enable_analyzer"] is True
        assert content.params["enable_match"] is True
        assert content.params["max_length"] == 65535

    def test_the_generated_column_carries_a_schema_level_bm25_function(
        self, build_call_helper
    ):
        schema = build_call_helper(DDL).kwargs["schema"]
        sparse = next(f for f in schema.fields if f.name == "content_sparse")
        assert sparse.dtype == DataType.SPARSE_FLOAT_VECTOR
        (function,) = schema.functions
        assert function.input_field_names == ["content"]
        assert function.output_field_names == ["content_sparse"]

    def test_the_short_as_spelling_works_too(self, build_call_helper):
        schema = build_call_helper(
            "CREATE TABLE d (id BIGINT PRIMARY KEY, c TEXT, "
            "cs SPARSEVEC AS (BM25(c)))"
        ).kwargs["schema"]
        (function,) = schema.functions
        assert function.output_field_names == ["cs"]


class TestSearch:
    def test_order_by_bm25_score_is_a_search_with_the_query_text(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT id, content FROM docs "
            "ORDER BY BM25_SCORE(content_sparse, :q) DESC LIMIT 5",
            {"q": "vector databases"},
        )
        assert call.method == "search"
        assert call.kwargs["anns_field"] == "content_sparse"
        assert call.kwargs["data"] == ["vector databases"]
        assert call.kwargs["search_params"]["metric_type"] == "BM25"
        assert call.kwargs["limit"] == 5

    def test_match_against_renders_as_a_text_match_filter(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT id FROM docs WHERE MATCH(content) AGAINST (:q) LIMIT 10",
            {"q": "hello world"},
        )
        assert call.method == "query"
        assert call.kwargs["filter"] == 'TEXT_MATCH(content, "hello world")'
