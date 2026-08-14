"""`IngestItemWorkflow`: orchestrates one Milvus insert through
`activities.insert_item`, with retries and the idempotency key that
makes those retries safe (see `activities.py`'s docstring)."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from models import IngestItemInput, InsertItemInput

# `activities.py` imports `milvusql`, which is not workflow-sandbox-safe
# to import directly at workflow module scope (it does real gRPC/network
# setup) -- `imports_passed_through` tells Temporal's sandbox to import
# it unmodified instead of re-executing it inside the sandbox. Standard
# pattern for sharing activity function references with a workflow file;
# see Temporal's own Python samples.
with workflow.unsafe.imports_passed_through():
    from activities import insert_item


@workflow.defn
class IngestItemWorkflow:
    @workflow.run
    async def run(self, item: IngestItemInput) -> int:
        # Generated once, deterministically, per workflow execution --
        # `workflow.uuid4()` (unlike stdlib `uuid.uuid4()`) is safe to
        # call from workflow code because it's derived from the
        # workflow's own replay-stable random seed, not real entropy:
        # every replay of this execution (including after a worker
        # crash and retry) produces the exact same value, which is
        # what makes reusing it as the idempotency key across
        # `insert_item` retries actually work.
        external_id = item.external_id or str(workflow.uuid4())
        return await workflow.execute_activity(
            insert_item,
            InsertItemInput(
                external_id=external_id,
                embedding=item.embedding,
                category=item.category,
                title=item.title,
            ),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=5,
            ),
        )
