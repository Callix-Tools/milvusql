"""Shared configuration, read once from the environment so
`schema.py`/`worker.py`/`run_workflow.py` all point at the same
targets."""

from __future__ import annotations

import os

#: Milvus Lite (a local file) by default, so the example runs with no
#: infrastructure beyond Temporal itself. Point at a real server with
#: `MILVUS_URI=http://localhost:19530 MILVUS_TOKEN=root:Milvus`.
MILVUS_URI = os.environ.get("MILVUS_URI", "./catalog_items.db")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")

TEMPORAL_ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
TASK_QUEUE = "milvusql-ingestion"
