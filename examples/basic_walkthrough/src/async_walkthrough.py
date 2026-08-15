"""The exact same tour as ``walkthrough.py``, over ``milvusql.aio``
instead of the sync PEP 249 surface -- schema setup included: unlike
Milvus Lite, a real server's async client (this example's target, via
``docker-compose.yaml`` -- see the README) has no gap around ``CREATE
INDEX``'s completion RPC, so this script stays all-async throughout,
with no sync client anywhere (see
``ast_to_pymilvus._build_create_index``'s own docstring in the root
``milvusql`` package for the Lite-only gap this would otherwise need
to work around). No explicit ``LOAD TABLE`` step either -- ``search``/
``query`` auto-load their target collection the first time a
connection touches it.

Run with:

    uv run src/async_walkthrough.py
"""

import asyncio
import logging

from const import (
    CATALOG,
    MILVUS_TOKEN,
    MILVUS_URI,
    QUERY_CREATE_INDEX,
    QUERY_CREATE_TABLE,
    QUERY_DELETE,
    QUERY_DROP_TABLE,
    QUERY_INSERT_DATA,
    QUERY_SELECT_ALL,
    QUERY_SELECT_PLAIN_FILTER,
    QUERY_SELECT_VECTOR_SEARCH,
    QUERY_UPDATE,
    QUERY_VECTOR,
)

from milvusql import aio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main() -> None:

    logger.info("\n--- connect (async) ---")
    conn = aio.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    cur = conn.cursor()

    try:
        logger.info("\n--- CREATE TABLE ---")
        await cur.execute(QUERY_DROP_TABLE)
        await cur.execute(QUERY_CREATE_TABLE)
        logger.info("collection 'walkthrough_items' created")

        logger.info("\n--- CREATE INDEX ---")
        await cur.execute(QUERY_CREATE_INDEX)
        logger.info("index built")

        logger.info("\n--- INSERT (batched) ---")
        await cur.executemany(
            QUERY_INSERT_DATA,
            [
                {"emb": emb, "cat": cat, "title": title}
                for emb, cat, title in CATALOG
            ],
        )
        logger.info(f"inserted {cur.rowcount} rows")

        logger.info("\n--- SELECT (plain filter) ---")
        await cur.execute(
            QUERY_SELECT_PLAIN_FILTER,
            {"cat": "book"},
        )
        for row in await cur.fetchall():
            logger.info(str(row))

        logger.info("\n--- SELECT (vector search) ---")
        await cur.execute(
            QUERY_SELECT_VECTOR_SEARCH,
            {"embed": QUERY_VECTOR},
        )
        async for row in cur:
            logger.info(str(row))

        logger.info("\n--- UPDATE ---")
        await cur.execute(
            QUERY_UPDATE,
            {"new": "sci-fi-book", "title": "Dune"},
        )
        logger.info(f"updated {cur.rowcount} row(s)")

        logger.info("\n--- DELETE ---")
        await cur.execute(
            QUERY_DELETE,
            {"cat": "movie"},
        )
        logger.info(f"deleted {cur.rowcount} row(s)")

        logger.info("\n--- final state ---")
        await cur.execute(QUERY_SELECT_ALL)
        for row in await cur.fetchall():
            logger.info(str(row))
    finally:
        logger.info("\n--- cleanup ---")
        await cur.execute(QUERY_DROP_TABLE)
        await conn.close()
        logger.info("connection closed")


if __name__ == "__main__":
    asyncio.run(main())
