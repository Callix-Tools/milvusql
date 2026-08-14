# pydantic-ai catalog agent

A [pydantic-ai](https://ai.pydantic.dev) shopping-assistant agent whose only
access to the product catalog is through two typed tools —
`search_products` (a real vector search over `milvusql-sqlalchemy`) and
`get_product` (an exact lookup by id). The model never sees raw SQL or the
database directly, only these tools and their `ProductResult` responses —
so its answers are grounded in whatever the catalog actually contains.

## Layout

- `catalog/config.py` — settings from the environment.
- `catalog/db.py` — the `products` collection (Core `Table`), and schema
  bootstrap.
- `catalog/embeddings.py` — pluggable text embedder: `all-MiniLM-L6-v2` by
  default, or a zero-dependency `deterministic` backend for smoke-testing.
- `catalog/seed.py` — bootstraps the schema and inserts a small sample
  catalog.
- `catalog/agent.py` — the agent, its two tools, and the `ProductResult`
  type they return.
- `catalog/main.py` — an interactive CLI chat loop.

## Two things worth noticing in `catalog/db.py`/`catalog/main.py`

1. **Schema bootstrap uses a sync engine.** `CREATE INDEX` waits for
   completion via an RPC that Milvus Lite's async gRPC server doesn't
   implement — a Lite-only gap, not a general one (see
   `ast_to_pymilvus._build_create_index`'s docstring in the root `milvusql`
   package) — so `bootstrap_schema()` builds the collection through a plain
   `create_engine`, and only the agent's tools (`catalog/agent.py`) talk to
   it through `milvusql+aio`.

2. **`catalog/main.py` re-runs `bootstrap_schema()` on startup, not just
   `seed.py`.** Milvus Lite's "loaded" state lives in the process that
   loaded it, not in the on-disk file — so a fresh process (the chat CLI)
   reopening the same file needs its own `load_collection()` call even
   though `seed.py` already made one in its own process. `bootstrap_schema`
   guards every step, so calling it again is cheap when nothing's missing.
   A real (non-Lite) server keeps load state independently of any one
   client connection, so this is a no-op there — correct either way, which
   is why `search_products` itself doesn't have to special-case Lite at
   all.

## Run it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...   # or any provider pydantic-ai supports --
                                # see AGENT_MODEL below

python -m catalog.seed          # once: creates + indexes + seeds the catalog
python -m catalog.main          # chat
```

No Milvus server needed by default — `DATABASE_URL` defaults to Milvus Lite
(`milvusql+aio:///./catalog.db`). Point at a real server instead with:

```bash
export DATABASE_URL="milvusql+aio://root:Milvus@localhost:19530/default"
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
EMBEDDING_BACKEND=deterministic python -m catalog.seed
```

## Example session

```
$ python -m catalog.main
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
