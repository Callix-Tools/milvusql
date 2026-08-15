# FastAPI image search

An image-search service on top of `milvusql-sqlalchemy`: upload images with
a caption, then search either by a query image or by a free-text
description (cross-modal search, via CLIP's shared image/text embedding
space).

## Layout

- `src/config.py` — settings for this example (a real Milvus target, the
  embedding backend).
- `src/db.py` — the `images` collection (Core `Table`), and schema
  bootstrap.
- `src/embeddings.py` — pluggable embedder: CLIP by default, or a
  zero-dependency `deterministic` backend for smoke-testing.
- `src/schemas.py` — response models.
- `src/main.py` — the FastAPI app and its routes.

This example has its own `pyproject.toml` but is deliberately **not** a
member of the repo's root `uv` workspace (unlike
[`examples/*`](../../../../examples)): its web/ML dependencies (`torch`,
`fastapi`, ...) are heavy enough that folding it into the monorepo's shared
`uv.lock`/`task install` would slow every contributor's sync and every
`ci-*.yml` job down for something none of them need. `[tool.uv.sources]` in
its `pyproject.toml` points at the local `milvusql-sqlalchemy`/`milvusql`
checkouts directly instead.

## Why schema bootstrap doesn't need a sync-client workaround

`db.py`'s `bootstrap_schema()` creates, indexes, and *doesn't* explicitly
load the collection, entirely through the async (`milvusql+aio`) engine.
Two things make that possible against this example's real Milvus target
(brought up by the bundled `docker-compose.yaml`, not Milvus Lite):

- `CREATE INDEX` waits for completion via an RPC that only Milvus Lite's
  async gRPC server fails to implement — a Lite-only gap, not a general one
  (see `ast_to_pymilvus._build_create_index`'s docstring in the root
  `milvusql` package) — so it just works here.
- `search`/`query` now auto-load their target collection the first time a
  connection touches it, so there's no `LOAD TABLE`/`load_collection()` step
  to work around at all.

## Run it

### 1. Start Milvus

```bash
docker compose up -d
```

Brings up `etcd`/`minio`/`milvus-standalone`/`attu` (the shared infra from
[`example_infra/`](../../../../example_infra) — the same containers the
root package's own examples use, under the shared Compose project
`milvus-examples`, so if you already have one of those running, this is a
no-op). Wait for it to report healthy (`docker compose ps`, ~90s on a cold
start) before continuing.

### 2. Run the app

```bash
uv run uvicorn main:app --app-dir src --reload
```

The first request that needs the CLIP model downloads its weights (a few
hundred MB, cached afterwards by `sentence-transformers`). To try the API
without that download, run with the zero-dependency embedder instead — it
exercises the exact same insert/search code paths, just with a
pseudo-embedding instead of a real one:

```bash
EMBEDDING_BACKEND=deterministic uv run uvicorn main:app --app-dir src --reload
```

### 3. Try it

```bash
# Insert
curl -F "file=@cat.jpg" -F "caption=a orange tabby cat" http://localhost:8000/images
curl -F "file=@dog.jpg" -F "caption=a golden retriever" http://localhost:8000/images

# Search by image
curl -F "file=@query.jpg" "http://localhost:8000/search/image?limit=5"

# Search by text (cross-modal -- only meaningful with EMBEDDING_BACKEND=clip)
curl "http://localhost:8000/search/text?q=a+dog+running+outside"

# Fetch / delete
curl http://localhost:8000/images/1
curl -X DELETE http://localhost:8000/images/1
```

Interactive docs: http://localhost:8000/docs

### 4. Tear down

```bash
docker compose down       # stop the containers, keep data volumes
docker compose down -v    # also wipe them
```
