"""Error-path coverage for ``MilvusDialect._assert_and_set_isolation_level``
(``act.py``'s own fix): an isolation_level/consistency-level value outside
Milvus's five real consistency levels is rejected up front by SQLAlchemy,
with a clear ``ArgumentError``, rather than being forwarded to ``pymilvus``
to fail with a less helpful ``InvalidConsistencyLevel``."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import ArgumentError

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]


def test_unknown_isolation_level_raises_argument_error(seeded_engine):
    engine, _items = seeded_engine
    with (
        pytest.raises(ArgumentError, match="Bounded"),
        engine.connect() as conn,
    ):
        conn.execution_options(isolation_level="NOT_A_REAL_LEVEL")
