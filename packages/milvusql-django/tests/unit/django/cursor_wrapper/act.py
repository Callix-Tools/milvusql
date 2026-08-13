"""Unit coverage for ``CursorWrapper``: translating Django's
``%s``/``%(name)s`` placeholder text into MilvusQL's ``:name`` binds,
then delegating to the wrapped cursor. Neither needs
``settings.configure()``/``django.setup()`` -- ``CursorWrapper`` only
wraps whatever cursor it's given."""

from __future__ import annotations

import typing as t

import pytest
from milvusql_django.base import CursorWrapper

import milvusql.dbapi as Database  # noqa: N812 -- matches base.py's own alias

pytestmark = [pytest.mark.unit, pytest.mark.django]


class _FakeCursor:
    """Records what it was called with -- a stub, not a real DBAPI
    cursor, since ``CursorWrapper`` only needs ``.execute``/
    ``.executemany`` to delegate to."""

    def __init__(self):
        self.calls = []
        self.rowcount = 0

    def execute(self, query, params):
        self.calls.append(("execute", query, params))
        return "executed"

    def executemany(self, query, param_list):
        self.calls.append(("executemany", query, param_list))
        return "executed_many"


class _FakeWrapper:
    """Stands in for ``DatabaseWrapper`` -- ``CursorWrapper`` only
    ever calls ``_table_needs_pad_vector`` on it (to decide whether an
    ``INSERT`` needs ``schema.PAD_VECTOR_FIELD`` spliced in). None of
    these tests target a padded table."""

    def _table_needs_pad_vector(self, table: str) -> bool:
        return False


def _wrap(cursor: _FakeCursor) -> CursorWrapper:
    """``CursorWrapper`` only ever calls ``.execute``/``.executemany``
    on its wrapped cursor, or delegates an unknown attribute straight
    through -- ``_FakeCursor`` satisfies that shape without being a
    real ``milvusql.dbapi.Cursor``. Same kind of understood cast
    ``test_translate.py`` uses for ``MilvusClient``."""
    return CursorWrapper(
        t.cast(Database.Cursor, cursor), t.cast("t.Any", _FakeWrapper())
    )


class TestConvert:
    def test_no_params_returns_the_query_unchanged_with_an_empty_dict(self):
        wrapper = _wrap(_FakeCursor())
        assert wrapper._convert("SELECT 1", None) == ("SELECT 1", {})

    def test_positional_params_become_param_n_named_binds_in_order(self):
        wrapper = _wrap(_FakeCursor())
        query, params = wrapper._convert(
            "SELECT * FROM t WHERE id = %s AND cat = %s", [1, "book"]
        )
        assert query == "SELECT * FROM t WHERE id = :param0 AND cat = :param1"
        assert params == {"param0": 1, "param1": "book"}

    def test_dict_params_become_colon_prefixed_binds_keyed_by_name(self):
        wrapper = _wrap(_FakeCursor())
        query, params = wrapper._convert(
            "SELECT * FROM t WHERE id = %(id)s AND cat = %(cat)s",
            {"id": 1, "cat": "book"},
        )
        assert query == "SELECT * FROM t WHERE id = :id AND cat = :cat"
        assert params == {"id": 1, "cat": "book"}

    def test_an_escaped_percent_s_is_left_untouched(self):
        wrapper = _wrap(_FakeCursor())
        query, params = wrapper._convert("...100%%s...WHERE id = %s", [1])
        assert query == "...100%%s...WHERE id = :param0"
        assert params == {"param0": 1}


class TestExecute:
    def test_execute_converts_then_delegates_to_the_wrapped_cursor(self):
        fake = _FakeCursor()
        wrapper = _wrap(fake)
        result = wrapper.execute("SELECT * FROM t WHERE id = %s", [1])
        assert result == "executed"
        assert fake.calls == [
            ("execute", "SELECT * FROM t WHERE id = :param0", {"param0": 1})
        ]

    def test_executemany_converts_each_param_set_independently(self):
        fake = _FakeCursor()
        wrapper = _wrap(fake)
        result = wrapper.executemany(
            "INSERT INTO t (id) VALUES (%s)", [[1], [2]]
        )
        assert result == "executed_many"
        assert fake.calls == [
            (
                "executemany",
                "INSERT INTO t (id) VALUES (:param0)",
                [{"param0": 1}, {"param0": 2}],
            )
        ]

    def test_executemany_with_an_empty_param_list_short_circuits(self):
        fake = _FakeCursor()
        wrapper = _wrap(fake)
        assert (
            wrapper.executemany("INSERT INTO t (id) VALUES (%s)", []) is None
        )
        assert fake.calls == []

    def test_unknown_attributes_delegate_to_the_wrapped_cursor(self):
        fake = _FakeCursor()
        fake.rowcount = 7
        wrapper = _wrap(fake)
        assert wrapper.rowcount == 7
