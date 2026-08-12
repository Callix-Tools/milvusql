"""Unit coverage for ``MilvusDialect``'s connection-state logic: the
isolation-level round-trip (D6's connection-level consistency-level
fallback), ``do_rollback``'s deliberate no-op (D7), and
``get_schema_names``'s constant. The reflection methods
(``get_columns``/``get_pk_constraint``/``get_indexes``/...) all need a
real ``pymilvus`` client behind a ``Connection`` and are covered end to
end in ``tests/integration/sqlalchemy/engine_roundtrip`` instead."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.dialect import MilvusDialect

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]


class _StubConnection:
    """Stands in for a ``milvusql.dbapi.Connection``: the isolation
    methods only ever touch a plain ``.consistency_level`` attribute."""

    consistency_level = None


@pytest.fixture
def dialect() -> MilvusDialect:
    return MilvusDialect()


class TestIsolationLevel:
    def test_defaults_to_bounded_when_unset(self, dialect):
        assert dialect.get_isolation_level(_StubConnection()) == "Bounded"

    def test_set_isolation_level_round_trips_through_get(self, dialect):
        conn = _StubConnection()
        dialect.set_isolation_level(conn, "Strong")
        assert conn.consistency_level == "Strong"
        assert dialect.get_isolation_level(conn) == "Strong"


def test_do_rollback_is_a_noop(dialect):
    """D7: SQLAlchemy calls this as routine pool bookkeeping, not on a
    user's explicit rollback -- it must always succeed silently."""
    assert dialect.do_rollback(_StubConnection()) is None


def test_get_schema_names_always_returns_default(dialect):
    assert dialect.get_schema_names(None) == ["default"]
