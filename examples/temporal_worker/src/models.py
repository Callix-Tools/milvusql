"""Dataclasses shared between `workflows.py` and `activities.py`.

Kept import-light (stdlib only): `workflows.py` imports this module
directly from inside Temporal's workflow sandbox, which restricts
what a workflow file can safely import at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InsertItemInput:
    """`insert_item` activity input -- exactly what it needs to run
    the idempotent insert, no more."""

    external_id: str
    embedding: list[float]
    category: str
    title: str


@dataclass
class IngestItemInput:
    """`IngestItemWorkflow` input. `external_id` is optional -- when
    omitted, the workflow generates one deterministically (see its
    own docstring) so retries of one workflow execution stay
    idempotent without the caller having to manage ids itself. Pass
    one explicitly to make re-running the *same logical* ingestion
    (e.g. reprocessing a known catalog id) safe across separate
    workflow executions too."""

    embedding: list[float]
    category: str
    title: str
    external_id: str | None = None
