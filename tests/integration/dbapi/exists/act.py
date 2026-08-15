"""Integration coverage for correlated ``[NOT] EXISTS`` (the
SQLAlchemy ``.any()``/``.has()`` and Django ``Exists(OuterRef)``
shape) against a real Milvus server."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]


@pytest.fixture
def parent_child(conn):
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE parents (id BIGINT PRIMARY KEY, name VARCHAR(32), "
        "v VECTOR(2)) WITH (consistency_level='Strong')"
    )
    cur.execute(
        "CREATE INDEX idx_pv ON parents (v) USING FLAT WITH (metric_type='L2')"
    )
    cur.execute(
        "CREATE TABLE children (id BIGINT PRIMARY KEY, parent_id BIGINT, "
        "active BOOLEAN, v VECTOR(2)) WITH (consistency_level='Strong')"
    )
    cur.execute(
        "CREATE INDEX idx_cv ON children (v) USING FLAT "
        "WITH (metric_type='L2')"
    )
    cur.executemany(
        "INSERT INTO parents (id, name, v) VALUES (:id, :name, :v)",
        [
            {"id": 1, "name": "with-active", "v": [0.1, 0.1]},
            {"id": 2, "name": "with-inactive", "v": [0.2, 0.2]},
            {"id": 3, "name": "childless", "v": [0.3, 0.3]},
        ],
    )
    cur.executemany(
        "INSERT INTO children (id, parent_id, active, v) "
        "VALUES (:id, :pid, :active, :v)",
        [
            {"id": 10, "pid": 1, "active": True, "v": [0.0, 0.0]},
            {"id": 11, "pid": 2, "active": False, "v": [0.0, 0.0]},
        ],
    )
    return cur


def test_exists_with_an_inner_filter(parent_child):
    parent_child.execute(
        "SELECT parents.name FROM parents WHERE EXISTS "
        "(SELECT 1 FROM children WHERE parents.id = children.parent_id "
        "AND children.active = true)"
    )
    assert parent_child.fetchall() == [("with-active",)]


def test_not_exists_is_the_anti_join(parent_child):
    parent_child.execute(
        "SELECT parents.name FROM parents WHERE NOT EXISTS "
        "(SELECT 1 FROM children WHERE parents.id = children.parent_id)"
    )
    assert parent_child.fetchall() == [("childless",)]


async def test_async_cursor_runs_the_same_decorrelation(aconn, parent_child):
    cur = aconn.cursor()
    await cur.execute(
        "SELECT parents.name FROM parents WHERE EXISTS "
        "(SELECT 1 FROM children WHERE parents.id = children.parent_id)"
    )
    names = {row[0] for row in await cur.fetchall()}
    assert names == {"with-active", "with-inactive"}
