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

`src/activities.py`'s `insert_item` closes that gap the only way Milvus allows:
the *workflow* generates a stable idempotency key (`external_id`) once, and
the activity checks for a row with that key before inserting. This isn't
atomic — a genuine race between two truly concurrent callers could still
double-insert — but Temporal never runs two attempts of the *same* activity
execution concurrently, so it closes the specific gap that matters here
(retry-after-crash), which a bare retry loop around `cur.execute()` would
not.

## Layout

- `src/schema.py` — one-time bootstrap: `CREATE TABLE`/`CREATE INDEX`/`LOAD
  TABLE` for the `catalog_items` collection. Run once before starting the
  worker.
- `src/activities.py` — `insert_item`: the idempotent insert described above.
  Opens one `milvusql` connection per worker **process** (not per activity
  call — see its docstring) and reuses it.
- `src/workflows.py` — `IngestItemWorkflow`: generates the idempotency key,
  calls `insert_item` with a retry policy, returns the inserted row's id.
- `src/worker.py` — connects to Temporal and runs the worker.
- `src/run_workflow.py` — starts one `IngestItemWorkflow` execution from the
  command line, as a stand-in for whatever triggers ingestion in a real
  pipeline (an API request, a queue message, a batch job).

## Run it

### 1. Start Milvus + Temporal

`src/config.py` hardcodes `MILVUS_URI = "http://localhost:19530"` and
`MILVUS_TOKEN = "root:Milvus"` — a real Milvus server, not Milvus Lite, and
no env var override. `docker-compose.yaml` brings up everything this example
needs in one shot:

```bash
docker compose up -d
```

That's `etcd`/`minio`/`milvus-standalone`/`attu` (the shared Milvus infra
from [`example_infra/milvus.compose.yml`](../../example_infra/milvus.compose.yml)
— the same containers [`basic_walkthrough/`](../basic_walkthrough) uses,
under the shared Compose project `milvus-examples`, so bringing either
example's stack up is enough for both) plus a `temporal` container running
Temporal's dev server (gRPC on `7233`, Web UI on
[`http://localhost:8233`](http://localhost:8233)). The dev server keeps its
state in memory on purpose — this is a throwaway demo namespace, not
something you need surviving a restart — so there's no `--db-filename`/volume
for it. `TEMPORAL_ADDRESS` in `src/config.py` defaults to `localhost:7233`
and is the one thing here that *does* read from the environment, if you
point at a Temporal server running elsewhere.

Wait for `milvus-standalone` to report healthy (`docker compose ps`, ~90s on
a cold start) before continuing.

### 2. Run the example

```bash
# once: create, index, and load the catalog_items collection
uv run src/schema.py

# keep running in its own terminal: the worker
uv run src/worker.py

# in another terminal: trigger one ingestion
uv run src/run_workflow.py --category book --title "Dune" \
    --embedding 0.1 0.12 0.11 0.09 0.1 0.13 0.11 0.1
```

`src/run_workflow.py` prints the row id `IngestItemWorkflow` returns. Re-running
it with the same `--external-id` (pass one explicitly, or reuse the id
printed by a failed run) is safe — that's the idempotency guard above at
work. Watch the run unfold in Temporal's Web UI at `http://localhost:8233`.

### 3. Tear down

```bash
docker compose down       # stop the containers, keep data volumes
docker compose down -v    # also wipe them
```
