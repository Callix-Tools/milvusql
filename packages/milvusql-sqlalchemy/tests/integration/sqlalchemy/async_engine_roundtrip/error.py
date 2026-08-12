"""Error-path integration coverage for ``MilvusDialect_aio``."""

from __future__ import annotations

import pytest
from sqlalchemy import column, select, table

from milvusql.dbapi.errors import ProgrammingError

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]


async def test_missing_collection_raises_programming_error(seeded_engine_aio):
    engine, _items = seeded_engine_aio
    stmt = select(column("id")).select_from(table("does_not_exist")).limit(1)
    async with engine.connect() as conn:
        with pytest.raises(Exception) as exc_info:
            await conn.execute(stmt)
    assert isinstance(exc_info.value.__cause__, ProgrammingError)
