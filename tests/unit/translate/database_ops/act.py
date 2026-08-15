"""Unit coverage for database-level DDL (``CREATE``/``DROP DATABASE``,
``USE``) and ``DROP INDEX`` -- each one pymilvus call, no rows back."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_create_database(build_call_helper):
    call = build_call_helper("CREATE DATABASE analytics")
    assert call.method == "create_database"
    assert call.kwargs == {"db_name": "analytics"}
    assert call.postprocess(None) == ([], None, -1, None)


def test_drop_database(build_call_helper):
    call = build_call_helper("DROP DATABASE analytics")
    assert call.method == "drop_database"
    assert call.kwargs == {"db_name": "analytics"}


def test_use_database(build_call_helper):
    call = build_call_helper("USE analytics")
    assert call.method == "use_database"
    assert call.kwargs == {"db_name": "analytics"}


def test_drop_index_names_collection_and_index(build_call_helper):
    call = build_call_helper("DROP INDEX idx_embedding ON items")
    assert call.method == "drop_index"
    assert call.kwargs == {
        "collection_name": "items",
        "index_name": "idx_embedding",
    }
