"""The activity that actually writes to Milvus.

Two production concerns this handles that a bare `cur.execute()` call
wouldn't:

* **One connection per worker process, not per activity call.**
  `_get_connection()` opens `milvusql.aio`'s connection lazily and
  caches it at module scope -- opening a fresh gRPC channel on every
  activity invocation would be both slow and defeat pymilvus's own
  connection management. Safe because a Temporal worker process runs
  activities cooperatively on one event loop, not across processes.

* **Idempotency.** Milvus has no unique constraint and no transaction
  to roll back (`milvusql.dbapi.Connection.rollback()` raises
  `NotSupportedError` on purpose). Temporal's activities are
  at-least-once: if `insert_item` completes its `INSERT` but the
  worker crashes before that's recorded, Temporal retries the *same*
  activity execution, which would otherwise insert the row twice.
  `insert_item` closes that gap with the `external_id` its caller
  (`IngestItemWorkflow`) generates once per workflow execution and
  reuses across retries: check for a row with that id first, insert
  only if none exists. Not atomic against genuinely concurrent
  callers -- but Temporal never runs two attempts of one activity
  execution concurrently, so it closes the retry-after-crash gap that
  actually matters here.
"""

from __future__ import annotations

from temporalio import activity

from config import MILVUS_TOKEN, MILVUS_URI
from milvusql import aio
from models import InsertItemInput

_connection: aio.AsyncConnection | None = None


def _get_connection() -> aio.AsyncConnection:
    global _connection  # noqa: PLW0603 -- deliberate module-level cache, see docstring
    if _connection is None:
        _connection = aio.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    return _connection


@activity.defn
async def insert_item(item: InsertItemInput) -> int:
    conn = _get_connection()
    cur = conn.cursor()

    await cur.execute(
        "SELECT id FROM catalog_items WHERE external_id = :eid LIMIT 1",
        {"eid": item.external_id},
    )
    existing = await cur.fetchone()
    if existing is not None:
        activity.logger.info(
            "external_id=%s already ingested as id=%s, skipping insert",
            item.external_id,
            existing[0],
        )
        return existing[0]

    await cur.execute(
        "INSERT INTO catalog_items "
        "(external_id, embedding, category, title) "
        "VALUES (:eid, :emb, :cat, :title)",
        {
            "eid": item.external_id,
            "emb": item.embedding,
            "cat": item.category,
            "title": item.title,
        },
    )
    activity.logger.info(
        "inserted external_id=%s as id=%s", item.external_id, cur.lastrowid
    )
    return cur.lastrowid
