# Basic walkthrough

A single, narrated script that exercises the `milvusql` DBAPI end to end
against a real Milvus server: connect, `CREATE TABLE`, `CREATE INDEX`, insert
rows, run a plain filter `SELECT`, run a vector search, `UPDATE`, `DELETE`,
close. No explicit `LOAD TABLE` step -- `search`/`query` auto-load their
target collection the first time a connection touches it.

Two versions of the exact same walkthrough:

- `walkthrough.py` — sync (`milvusql.connect`)
- `async_walkthrough.py` — async (`milvusql.aio.connect`); schema setup
  (`CREATE TABLE`/`CREATE INDEX`) still runs through the *sync* client even
  here -- see that script's own docstring for why.

## 1. Start Milvus

This example targets a real Milvus server, not Milvus Lite: `src/const.py`
hardcodes `MILVUS_URI = "http://localhost:19530"` and
`MILVUS_TOKEN = "root:Milvus"`, with no env var override. Bring a server up
with the bundled `docker-compose.yaml`:

```bash
docker compose up -d
```

That starts four containers, all on a private `milvus` network:

- `etcd` — Milvus's metadata store
- `minio` — its object storage backend
- `milvus-standalone` — the Milvus server itself, exposed on `19530`
  (the port `const.py` connects to) and `9091` (health/metrics)
- `attu` — a web UI at `http://localhost:8000` for browsing the collection
  this script creates, if you want to look around by hand

`milvus-standalone` depends on `etcd`/`minio` being healthy first and can
take up to ~90s to come up on a cold start. Wait for it before running the
script:

```bash
docker compose ps   # watch until milvus-standalone shows "healthy"
```

`root:Milvus` is Milvus's default superuser credential, so no extra auth
setup is needed to match what `const.py` sends.

## 2. Run it

```bash
uv run src/walkthrough.py
uv run src/async_walkthrough.py
```

Each run: drops `walkthrough_items` if a previous run left it behind,
creates it fresh, builds an HNSW index on its embedding column, inserts a
4-row catalog (books + movies) in one batch, runs a plain-filter `SELECT`
(`category = 'book'`), runs a vector search against a query embedding,
`UPDATE`s one row's category, `DELETE`s the rows in one category, prints the
final state, then drops the collection again -- so it's safe to re-run
repeatedly against the same server.

## 3. Tear down

```bash
docker compose down       # stop the containers, keep etcd/minio/milvus data
docker compose down -v    # also wipe the data volumes
```
