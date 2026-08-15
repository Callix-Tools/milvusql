"""Runs the Temporal worker: polls `TASK_QUEUE` for
`IngestItemWorkflow` executions and `insert_item` activity tasks.

    python worker.py

Requires the `catalog_items` collection to already exist --
run `schema.py` once first.
"""

from __future__ import annotations

import asyncio
import logging

from activities import insert_item
from config import TASK_QUEUE, TEMPORAL_ADDRESS
from temporalio.client import Client
from temporalio.worker import Worker
from workflows import IngestItemWorkflow

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    client = await Client.connect(TEMPORAL_ADDRESS)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[IngestItemWorkflow],
        activities=[insert_item],
    )
    print(
        f"worker listening on {TEMPORAL_ADDRESS!r}, "
        f"task queue {TASK_QUEUE!r} ... (Ctrl-C to stop)"
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
