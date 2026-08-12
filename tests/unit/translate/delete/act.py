"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on
``DELETE``."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestDelete:
    def test_with_filter_renders_the_where_clause(self, build_call_helper):
        call = build_call_helper(
            "DELETE FROM items WHERE category = :cat", {"cat": "movie"}
        )
        assert call.method == "delete"
        assert call.kwargs == {
            "collection_name": "items",
            "filter": 'category == "movie"',
        }

    def test_without_filter_omits_the_filter_kwarg(self, build_call_helper):
        call = build_call_helper("DELETE FROM items")
        assert call.kwargs == {"collection_name": "items"}

    def test_postprocess_reads_delete_count_from_a_dict(
        self, build_call_helper
    ):
        call = build_call_helper("DELETE FROM items")
        assert call.postprocess({"delete_count": 3}) == ([], None, 3, None)

    def test_postprocess_falls_back_to_list_length_for_older_servers(
        self, build_call_helper
    ):
        """``MilvusClient.delete``'s own compatibility branch: older
        servers return a bare list of deleted primary keys instead of
        ``{"delete_count": n}`` (confirmed directly against Milvus
        Lite -- see ``ast_to_pymilvus._mutation_count``)."""
        call = build_call_helper("DELETE FROM items")
        assert call.postprocess([1, 2, 3]) == ([], None, 3, None)
