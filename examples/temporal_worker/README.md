# Temporal worker: durable Milvus ingestion

A [Temporal](https://temporal.io) workflow + activity that inserts a row
into Milvus through `milvusql` — the pattern for a production ingestion
pipeline where writes must survive worker crashes/restarts, need automatic
retries with backoff, and want ingestion progress visible in Temporal's own
UI/CLI, instead of a bare script that dies (and loses its place) the moment
the process does.

## Why this needs an idempotency guard

Milvus has neither a transaction/rollback (`Connection.rollback()` in this
package raises `NotSupportedError` on purpose — see its docstring) nor a
unique constraint to lean on. Temporal's own guarantee is **at-least-once**
activity execution: if `insert_item` completes the `INSERT` but the worker
crashes (or the response is lost) before Temporal records the activity as
done, Temporal retries it — which, without a guard, means the same row gets
inserted twice.

`activities.py`'s `insert_item` closes that gap the only way Milvus allows:
the *workflow* generates a stable idempotency key (`external_id`) once, and
the activity checks for a row with that key before inserting. This isn't
atomic — a genuine race between two truly concurrent callers could still
double-insert — but Temporal never runs two attempts of the *same* activity
execution concurrently, so it closes the specific gap that matters here
(retry-after-crash), which a bare retry loop around `cur.execute()` would
not.

## Layout

- `schema.py` — one-time bootstrap: `CREATE TABLE`/`CREATE INDEX`/`LOAD
  TABLE` for the `catalog_items` collection. Run once before starting the
  worker.
- `activities.py` — `insert_item`: the idempotent insert described above.
  Opens one `milvusql` connection per worker **process** (not per activity
  call — see its docstring) and reuses it.
- `workflows.py` — `IngestItemWorkflow`: generates the idempotency key,
  calls `insert_item` with a retry policy, returns the inserted row's id.
- `worker.py` — connects to Temporal and runs the worker.
- `run_workflow.py` — starts one `IngestItemWorkflow` execution from the
  command line, as a stand-in for whatever triggers ingestion in a real
  pipeline (an API request, a queue message, a batch job).

## Run it

Requires a Temporal server (the [dev
server](https://docs.temporal.io/cli#start-dev-server) is enough locally)
and a Milvus target — Milvus Lite (a local file, the default below) or a
real server via `MILVUS_URI`.

```bash
pip install -r requirements.txt

# Terminal 1: Temporal dev server
temporal server start-dev

# Terminal 2: create the collection once
python schema.py

# Terminal 3: the worker (keep running)
python worker.py

# Terminal 4: trigger ingestion
python run_workflow.py --category book --title "Dune" \
    --embedding 0.1 0.12 0.11 0.09 0.1 0.13 0.11 0.1
```

`run_workflow.py` prints the row id `IngestItemWorkflow` returns. Re-running
it with the same `--external-id` (pass one explicitly, or reuse the id
printed by a failed run) is safe — that's the idempotency guard above at
work.
