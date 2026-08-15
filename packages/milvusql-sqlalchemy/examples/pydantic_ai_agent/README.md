# pydantic-ai catalog agent

A [pydantic-ai](https://ai.pydantic.dev) shopping-assistant agent whose only
access to the product catalog is through two typed tools —
`search_products` (a real vector search over `milvusql-sqlalchemy`) and
`get_product` (an exact lookup by id). The model never sees raw SQL or the
database directly, only these tools and their `ProductResult` responses —
so its answers are grounded in whatever the catalog actually contains.

## Layout

- `src/config.py` — settings for this example (a real Milvus target, the
  agent model, the embedding backend).
- `src/db.py` — the `products` collection (Core `Table`), and schema
  bootstrap.
- `src/embeddings.py` — pluggable text embedder: `all-MiniLM-L6-v2` by
  default, or a zero-dependency `deterministic` backend for smoke-testing.
- `src/seed.py` — bootstraps the schema and inserts a small sample catalog.
- `src/agent.py` — the agent, its two tools, and the `ProductResult` type
  they return.
- `src/main.py` — an interactive CLI chat loop.

This example has its own `pyproject.toml` but is deliberately **not** a
member of the repo's root `uv` workspace (unlike
[`examples/*`](../../../../examples)): its agent/ML dependencies (`torch`,
`pydantic-ai`, ...) are heavy enough that folding it into the monorepo's
shared `uv.lock`/`task install` would slow every contributor's sync and
every `ci-*.yml` job down for something none of them need. `[tool.uv.sources]`
in its `pyproject.toml` points at the local `milvusql-sqlalchemy`/`milvusql`
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
  to work around at all — including for `main.py`'s chat loop, run as a
  fresh process after `seed.py` already exited: against a real server, load
  state isn't tied to the process that set it, and auto-load would cover it
  either way.

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

### 2. Run the example

```bash
export OPENAI_API_KEY=sk-...   # or any provider pydantic-ai supports --
                                # see AGENT_MODEL below

uv run src/seed.py    # once: creates + indexes + seeds the catalog
uv run src/main.py    # chat
```

Use a different model provider with `AGENT_MODEL` (any string
[pydantic-ai accepts](https://ai.pydantic.dev/models/), e.g.
`AGENT_MODEL="anthropic:claude-sonnet-4-5"` with `ANTHROPIC_API_KEY` set).

To try the agent's tool-calling without an LLM API key or the embedding
download, run seeding and chat with the zero-dependency embedder — it
exercises the exact same search/lookup code paths, just with a
pseudo-embedding instead of real semantics, so `search_products` results
won't be meaningfully ranked:

```bash
EMBEDDING_BACKEND=deterministic uv run src/seed.py
```

## Example session

```
$ uv run src/main.py
Catalog assistant -- ask about products ('quit' to exit).
> I need something warm for cold hikes
I found a couple of good options for cold-weather hikes:

1. **ThermaCore Fleece Jacket** ($74.00) — a warm midweight fleece
   jacket built for cold-weather layering.
2. **Trailblazer 2 Hiking Boots** ($129.99) — waterproof leather
   hiking boots with ankle support for cold, rocky trails.

Want more detail on either one?
> tell me more about the first one
```

### 3. Tear down

```bash
docker compose down       # stop the containers, keep data volumes
docker compose down -v    # also wipe them
```
