# Basic walkthrough

A single, narrated script that exercises the `milvusql` DBAPI end to end:
connect, `CREATE TABLE`, `CREATE INDEX`, `LOAD TABLE`, insert rows, run a
plain filter `SELECT`, run a vector search, `UPDATE`, `DELETE`, close.

Two versions of the exact same walkthrough:

- `walkthrough.py` — sync (`milvusql.connect`)
- `async_walkthrough.py` — async (`milvusql.aio.connect`)

## Run it

No Milvus server needed — both scripts point at
[Milvus Lite](https://milvus.io/docs/milvus_lite.md) (a local on-disk file),
so there's nothing to install beyond the Python package:

```bash
pip install milvusql
python walkthrough.py
python async_walkthrough.py
```

To run against a real Milvus server instead, set `MILVUS_URI` (and
`MILVUS_TOKEN` if auth is enabled):

```bash
export MILVUS_URI="http://localhost:19530"
export MILVUS_TOKEN="root:Milvus"
python walkthrough.py
```

Each run creates its own collection (`walkthrough_items`) and drops it again
at the end, so it's safe to re-run repeatedly against the same server.
