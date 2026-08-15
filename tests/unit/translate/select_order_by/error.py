"""Error-path coverage for ``translate.ast_to_pymilvus.build_call`` on
a plain ``SELECT ... ORDER BY <scalar column>`` -- Milvus's per-call
row ceiling is a real correctness hazard here: sorting a truncated
subset of the matching rows would silently return the wrong "top N"
with no error at all."""

from __future__ import annotations

import pytest

from milvusql.translate._common import DEFAULT_QUERY_LIMIT

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_hitting_the_row_ceiling_pages_instead_of_sorting_a_partial_view(
    build_call_helper, drive_chain
):
    """A scalar ``ORDER BY`` fetch that comes back at Milvus's per-call
    ceiling used to raise; it now pages through the rest (primary-key
    cursor), so the sort runs over every matching row and the true top
    LIMIT comes back."""
    call = build_call_helper(
        "SELECT id, price FROM items ORDER BY price DESC LIMIT 2"
    )
    first = [{"id": i, "price": i} for i in range(DEFAULT_QUERY_LIMIT)]
    describe = {"fields": [{"name": "id", "is_primary": True}]}
    remainder = [{"id": DEFAULT_QUERY_LIMIT, "price": DEFAULT_QUERY_LIMIT}]
    calls, (rows, _d, _rc, _l) = drive_chain(
        call, [first, describe, first, remainder]
    )
    assert calls[1].method == "describe_collection"
    assert rows == [
        (DEFAULT_QUERY_LIMIT, DEFAULT_QUERY_LIMIT),
        (DEFAULT_QUERY_LIMIT - 1, DEFAULT_QUERY_LIMIT - 1),
    ]


def test_one_row_under_the_ceiling_still_sorts_normally(build_call_helper):
    """The ceiling check must not fire a row early -- exactly
    ``DEFAULT_QUERY_LIMIT - 1`` matching rows is a query Milvus's
    ``query()`` RPC can answer in full, not a truncated one."""
    call = build_call_helper(
        "SELECT id, price FROM items ORDER BY price DESC LIMIT 1"
    )
    raw = [{"id": i, "price": i} for i in range(DEFAULT_QUERY_LIMIT - 1)]
    rows, _description, rowcount, _lastrowid = call.postprocess(raw)
    assert rows == [(DEFAULT_QUERY_LIMIT - 2, DEFAULT_QUERY_LIMIT - 2)]
    assert rowcount == 1
