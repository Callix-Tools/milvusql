# FastAPI image search

An image-search service on top of `milvusql-sqlalchemy`: upload images with
a caption, then search either by a query image or by a free-text
description (cross-modal search, via CLIP's shared image/text embedding
space).

## Layout

- `app/config.py` — settings from the environment.
- `app/db.py` — the `images` collection (Core `Table`), and schema
  bootstrap.
- `app/embeddings.py` — pluggable embedder: CLIP by default, or a
  zero-dependency `deterministic` backend for smoke-testing.
- `app/schemas.py` — response models.
- `app/main.py` — the FastAPI app and its routes.

## Why schema bootstrap uses a sync engine

`app.db.bootstrap_schema()` creates, indexes, and loads the collection
through a plain (sync) `create_engine`, even though every request handler
in `app/main.py` is async. `CREATE INDEX` waits for completion via an RPC
that Milvus Lite's async gRPC server doesn't implement — a Lite-only gap,
not a general one (see `ast_to_pymilvus._build_create_index`'s docstring in
the root `milvusql` package) — so building the schema once through the sync
client and only switching to async for request handling sidesteps it
entirely. Same pattern the `milvusql-sqlalchemy` test suite itself uses
(`seeded_engine_aio`).

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

No Milvus server needed by default — `DATABASE_URL` defaults to Milvus Lite
(`milvusql+aio:///./image_search.db`). Point at a real server instead with:

```bash
export DATABASE_URL="milvusql+aio://root:Milvus@localhost:19530/default"
```

The first request that needs the CLIP model downloads its weights (a few
hundred MB, cached afterwards by `sentence-transformers`). To try the API
without that download, run with the zero-dependency embedder instead — it
exercises the exact same insert/search code paths, just with a
pseudo-embedding instead of a real one:

```bash
EMBEDDING_BACKEND=deterministic uvicorn app.main:app --reload
```

## Try it

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
