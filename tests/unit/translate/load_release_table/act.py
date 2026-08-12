"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on
``LOAD TABLE`` / ``RELEASE TABLE``."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestLoadReleaseTable:
    def test_load_table_maps_to_load_collection(self, build_call_helper):
        call = build_call_helper("LOAD TABLE items")
        assert call.method == "load_collection"
        assert call.kwargs == {"collection_name": "items"}

    def test_load_table_with_replicas_maps_to_replica_number(
        self, build_call_helper
    ):
        call = build_call_helper("LOAD TABLE items WITH (replicas=2)")
        assert call.kwargs == {"collection_name": "items", "replica_number": 2}

    def test_release_table_maps_to_release_collection(self, build_call_helper):
        call = build_call_helper("RELEASE TABLE items")
        assert call.method == "release_collection"
        assert call.kwargs == {"collection_name": "items"}
