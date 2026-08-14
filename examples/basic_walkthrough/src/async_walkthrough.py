"""The exact same tour as ``walkthrough.py``, over ``milvusql.aio``
instead of the sync PEP 249 surface.

Run with:

    python async_walkthrough.py

Defaults to a local Milvus Lite file (``./walkthrough_async.db``) --
different filename than ``walkthrough.py``'s so the two can be run one
after another without colliding. Point at a real server the same way:

    MILVUS_URI=http://localhost:19530 MILVUS_TOKEN=root:Milvus python async_walkthrough.py

One deliberate wrinkle: schema setup (``CREATE TABLE``/``CREATE
INDEX``) runs through the *sync* client, not ``milvusql.aio``, even
though this is otherwise an all-async script. ``CREATE INDEX`` waits
for completion via an ``AllocTimestamp`` RPC that Milvus Lite's async
gRPC server doesn't implement (a Lite-only gap -- a real server's
async client handles it fine) -- see
``ast_to_pymilvus._build_create_index``'s own docstring. Building the
schema through the sync client and only switching to async for
DML/DQL sidesteps that gap entirely, and is exactly the pattern this
project's own async test suite uses (``seeded_engine_aio`` in
``milvusql-sqlalchemy``'s fixtures). No explicit ``LOAD TABLE`` step
either way -- both clients auto-``LOAD`` a collection the first time
they run a search/query against it.
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
        logger.info("\n--- CREATE TABLE (sync) ---")
        await cur.execute(QUERY_DROP_TABLE)
        await cur.execute(QUERY_CREATE_TABLE)
        logger.info("collection 'walkthrough_items' created")

        logger.info("\n--- CREATE INDEX (sync) ---")
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
