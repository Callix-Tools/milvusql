"""The exact same tour as ``walkthrough.py``, over ``milvusql.aio``
instead of the sync PEP 249 surface.

Run with:

    python async_walkthrough.py

Defaults to a local Milvus Lite file (``./walkthrough_async.db``) --
different filename than ``walkthrough.py``'s so the two can be run one
after another without colliding. Point at a real server the same way:

    MILVUS_URI=http://localhost:19530 MILVUS_TOKEN=root:Milvus python async_walkthrough.py

One deliberate wrinkle: schema setup (``CREATE TABLE``/``CREATE
INDEX``/``LOAD TABLE``) runs through the *sync* client, not
``milvusql.aio``, even though this is otherwise an all-async script.
``CREATE INDEX`` waits for completion via an ``AllocTimestamp`` RPC
that Milvus Lite's async gRPC server doesn't implement (a Lite-only
gap -- a real server's async client handles it fine) -- see
``ast_to_pymilvus._build_create_index``'s own docstring. Building the
schema through the sync client and only switching to async for
DML/DQL sidesteps that gap entirely, and is exactly the pattern this
project's own async test suite uses (``seeded_engine_aio`` in
``milvusql-sqlalchemy``'s fixtures).
"""

from __future__ import annotations

import asyncio
import os

import milvusql
from milvusql import aio
from milvusql.dbapi import errors

MILVUS_URI = os.environ.get("MILVUS_URI", "./walkthrough_async.db")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")

CATALOG = [
    # (embedding, category, title)
    ([0.10, 0.12, 0.11, 0.09, 0.10, 0.13, 0.11, 0.10], "book", "Dune"),
    ([0.11, 0.10, 0.12, 0.10, 0.09, 0.11, 0.10, 0.12], "book", "Neuromancer"),
    ([0.91, 0.88, 0.90, 0.92, 0.89, 0.91, 0.90, 0.88], "movie", "Blade Runner"),
    ([0.90, 0.92, 0.89, 0.91, 0.90, 0.88, 0.91, 0.90], "movie", "The Matrix"),
]


def step(title: str) -> None:
    print(f"\n--- {title} ---")


def bootstrap_schema() -> None:
    """Sync connection, used once, only for DDL -- see the module
    docstring for why."""
    conn = milvusql.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    cur = conn.cursor()
    try:
        step("CREATE TABLE (sync)")
        try:
            cur.execute("DROP TABLE walkthrough_items")
        except milvusql.Error:
            pass
        cur.execute(
            "CREATE TABLE walkthrough_items ("
            "id BIGINT PRIMARY KEY AUTO_INCREMENT, "
            "embedding VECTOR(8), "
            "category VARCHAR(64), "
            "title VARCHAR(128)"
            ") WITH (shards=1, consistency_level='Strong')"
        )
        print("collection 'walkthrough_items' created")

        step("CREATE INDEX + LOAD TABLE (sync)")
        cur.execute(
            "CREATE INDEX idx_embedding ON walkthrough_items (embedding) "
            "USING HNSW WITH (metric_type='COSINE', M=16, ef_construction=200)"
        )
        cur.execute("LOAD TABLE walkthrough_items")
        print("index built, collection loaded")
    finally:
        conn.close()


async def main() -> None:
    bootstrap_schema()

    step("connect (async)")
    conn = aio.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    cur = conn.cursor()

    try:
        step("INSERT (batched)")
        await cur.executemany(
            "INSERT INTO walkthrough_items (embedding, category, title) "
            "VALUES (:emb, :cat, :title)",
            [
                {"emb": emb, "cat": cat, "title": title}
                for emb, cat, title in CATALOG
            ],
        )
        print(f"inserted {cur.rowcount} rows")

        step("SELECT (plain filter)")
        await cur.execute(
            "SELECT id, title FROM walkthrough_items "
            "WHERE category = :cat LIMIT 10",
            {"cat": "book"},
        )
        for row in await cur.fetchall():
            print(row)

        step("SELECT (vector search)")
        query_vector = [0.10, 0.11, 0.10, 0.10, 0.11, 0.12, 0.10, 0.11]
        await cur.execute(
            "SELECT id, title, category FROM walkthrough_items "
            "ORDER BY embedding <=> :q LIMIT 3",
            {"q": query_vector},
        )
        # `AsyncCursor` also supports `async for row in cur:` once a
        # query has been executed, as an alternative to `fetchall()`.
        async for row in cur:
            print(row)

        step("UPDATE")
        await cur.execute(
            "UPDATE walkthrough_items SET category = :new "
            "WHERE title = :title",
            {"new": "sci-fi-book", "title": "Dune"},
        )
        print(f"updated {cur.rowcount} row(s)")

        step("DELETE")
        await cur.execute(
            "DELETE FROM walkthrough_items WHERE category = :cat",
            {"cat": "movie"},
        )
        print(f"deleted {cur.rowcount} row(s)")

        step("final state")
        await cur.execute(
            "SELECT id, title, category FROM walkthrough_items"
        )
        for row in await cur.fetchall():
            print(row)
    finally:
        step("cleanup")
        try:
            await cur.execute("DROP TABLE walkthrough_items")
        except errors.Error:
            pass
        await conn.close()
        print("connection closed")


if __name__ == "__main__":
    asyncio.run(main())
