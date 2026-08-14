"""Shared configuration, read once from the environment so
`schema.py`/`worker.py`/`run_workflow.py` all point at the same
targets."""

from __future__ import annotations

import os


MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"  # noqa: S105

TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
TASK_QUEUE = "milvusql-ingestion"
