"""``DatabaseOperations`` -- unit coverage for the handful of hooks
Django's base SQL generation needs filled in to target MilvusQL text."""

from __future__ import annotations

import pytest
from milvusql_django.operations import DatabaseOperations

pytestmark = [pytest.mark.unit, pytest.mark.django]


@pytest.fixture
def ops():
    return DatabaseOperations(connection=None)


class TestQuoteName:
    def test_wraps_a_bare_name_in_double_quotes(self, ops):
        assert ops.quote_name("id") == '"id"'

    def test_an_already_quoted_name_is_left_unchanged(self, ops):
        assert ops.quote_name('"already"') == '"already"'


def test_no_limit_value_is_none(ops):
    """MilvusQL's ``LIMIT`` is simply omittable, unlike sqlite's
    ``-1`` sentinel."""
    assert ops.no_limit_value() is None


def test_sql_flush_deletes_every_table_with_no_filter(ops):
    assert ops.sql_flush(None, ["items", "other"]) == [
        'DELETE FROM "items";',
        'DELETE FROM "other";',
    ]


def test_last_insert_id_returns_the_cursors_lastrowid_verbatim(ops):
    cursor = type("FakeCursor", (), {"lastrowid": 42})()
    assert ops.last_insert_id(cursor, "items", "id") == 42
