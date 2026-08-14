"""Starts one `IngestItemWorkflow` execution -- a stand-in for whatever
triggers ingestion in a real pipeline (an API request, a queue
message, a batch job).

    python run_workflow.py --category book --title "Dune" \\
        --embedding 0.1 0.12 0.11 0.09 0.1 0.13 0.11 0.1

Pass `--external-id` to control the idempotency key explicitly (e.g.
to safely re-trigger ingestion of a known catalog id); omit it to let
the workflow generate one itself.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from config import TASK_QUEUE, TEMPORAL_ADDRESS
from models import IngestItemInput
from temporalio.client import Client
from workflows import IngestItemWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--embedding",
        required=True,
        nargs="+",
        type=float,
        help="Space-separated vector components, e.g. 0.1 0.2 0.3 ...",
    )
    parser.add_argument(
        "--external-id",
        default=None,
        help="Idempotency key; omit to let the workflow generate one",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    client = await Client.connect(TEMPORAL_ADDRESS)

    # Temporal deduplicates by workflow id -- deriving it from
    # `external_id` when one is given means re-running this script
    # with the same `--external-id` reuses/reconnects to the same
    # workflow execution instead of starting a new one.
    workflow_id = f"ingest-item-{args.external_id or uuid.uuid4()}"
    result = await client.execute_workflow(
        IngestItemWorkflow.run,
        IngestItemInput(
            embedding=args.embedding,
            category=args.category,
            title=args.title,
            external_id=args.external_id,
        ),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    print(f"inserted row id={result} (workflow_id={workflow_id!r})")


if __name__ == "__main__":
    asyncio.run(main())
