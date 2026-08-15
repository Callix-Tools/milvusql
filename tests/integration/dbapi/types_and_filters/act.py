"""Integration coverage for the README's column-type table and filter
functions against a real Milvus server: ``ARRAY<T>(n)``, ``JSON`` path
filters, the array/json filter-function roster, and the dimensioned
binary-vector spelling."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]


@pytest.fixture
def gadgets(conn):
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE gadgets ("
        "id BIGINT PRIMARY KEY, "
        "tags ARRAY<VARCHAR(32)>(20), "
        "nums ARRAY<BIGINT>(10), "
        "meta JSON, "
        "v VECTOR(2)"
        ") WITH (consistency_level='Strong')"
    )
    cur.execute(
        "CREATE INDEX idx_g ON gadgets (v) USING FLAT WITH (metric_type='L2')"
    )
    cur.executemany(
        "INSERT INTO gadgets (id, tags, nums, meta, v) "
        "VALUES (:id, :tags, :nums, :meta, :v)",
        [
            {
                "id": 1,
                "tags": ["red", "sale"],
                "nums": [1, 2, 3],
                "meta": {"brand": "acme", "spec": {"w": 5}},
                "v": [0.1, 0.1],
            },
            {
                "id": 2,
                "tags": ["blue"],
                "nums": [4],
                "meta": {"brand": "other", "spec": {"w": 20}},
                "v": [0.9, 0.9],
            },
        ],
    )
    return cur


def test_array_contains(gadgets):
    gadgets.execute(
        "SELECT id FROM gadgets WHERE ARRAY_CONTAINS(tags, :t)", {"t": "red"}
    )
    assert gadgets.fetchall() == [(1,)]


def test_array_contains_any_with_a_list_bind(gadgets):
    gadgets.execute(
        "SELECT id FROM gadgets WHERE ARRAY_CONTAINS_ANY(tags, :ts)",
        {"ts": ["blue", "green"]},
    )
    assert gadgets.fetchall() == [(2,)]


def test_array_length_comparison(gadgets):
    gadgets.execute(
        "SELECT id FROM gadgets WHERE ARRAY_LENGTH(nums) = :n", {"n": 3}
    )
    assert gadgets.fetchall() == [(1,)]


def test_json_path_filters(gadgets):
    gadgets.execute(
        "SELECT id FROM gadgets WHERE meta['brand'] = :b", {"b": "acme"}
    )
    assert gadgets.fetchall() == [(1,)]
    gadgets.execute(
        "SELECT id FROM gadgets WHERE meta['spec']['w'] > :w", {"w": 10}
    )
    assert gadgets.fetchall() == [(2,)]


def test_array_and_json_columns_round_trip(gadgets):
    gadgets.execute(
        "SELECT tags, nums, meta FROM gadgets WHERE id = :i", {"i": 1}
    )
    (row,) = gadgets.fetchall()
    assert row[0] == ["red", "sale"]
    assert row[1] == [1, 2, 3]
    assert row[2]["spec"]["w"] == 5


def test_describe_round_trips_the_array_spelling(gadgets):
    gadgets.execute("DESCRIBE gadgets")
    rows = {row[0]: row for row in gadgets.fetchall()}
    assert rows["tags"][1] == "ARRAY<VARCHAR>(20)"
    assert rows["nums"][1] == "ARRAY<BIGINT>(10)"
    assert rows["meta"][1] == "JSON"


def test_binary_vector_ddl_insert_and_round_trip(conn):
    """``BINARYVEC(dim)`` -- DDL maps to ``BINARY_VECTOR``, bind values
    pass through as ``bytes`` (dim/8 of them per row), and the stored
    vector reads back."""
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE bin_items (id BIGINT PRIMARY KEY, bv BINARYVEC(16)) "
        "WITH (consistency_level='Strong')"
    )
    cur.execute(
        "CREATE INDEX idx_bv ON bin_items (bv) USING BIN_FLAT "
        "WITH (metric_type='HAMMING')"
    )
    cur.execute(
        "INSERT INTO bin_items (id, bv) VALUES (:id, :bv)",
        {"id": 1, "bv": bytes([0b10101010, 0b01010101])},
    )
    cur.execute("DESCRIBE bin_items")
    rows = {row[0]: row for row in cur.fetchall()}
    assert rows["bv"][1] == "BINARYVEC(16)"
    cur.execute("SELECT id, bv FROM bin_items WHERE id = :i", {"i": 1})
    (row,) = cur.fetchall()
    assert row[0] == 1
    stored = row[1]
    # pymilvus hands a binary vector back as `bytes` (sometimes wrapped
    # in a one-element list, varying by server/client version) -- the
    # payload is what matters.
    if isinstance(stored, (list, tuple)):
        stored = stored[0]
    assert bytes(stored) == bytes([0b10101010, 0b01010101])
