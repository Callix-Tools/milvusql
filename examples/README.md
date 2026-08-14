# Examples

Small, runnable programs showing how to use the `milvusql` DBAPI outside of
the test suite. Each subdirectory is standalone: its own `README.md` and
`requirements.txt`, no shared code to chase across directories.

| Example | Shows |
| --- | --- |
| [`basic_walkthrough/`](basic_walkthrough) | A guided, top-to-bottom tour of the DBAPI: connect, `CREATE TABLE`/`CREATE INDEX`, insert, plain filter `SELECT`, vector search, `UPDATE`/`DELETE` — sync and async. Start here if you're new to `milvusql`. |
| [`temporal_worker/`](temporal_worker) | A [Temporal](https://temporal.io) workflow/activity that inserts rows into Milvus as a durable, retry-safe ingestion pipeline — including the idempotency guard a Milvus write needs to be safe under Temporal's at-least-once activity retries. |

See also the SQLAlchemy dialect's own examples in
[`packages/milvusql-sqlalchemy/examples/`](../packages/milvusql-sqlalchemy/examples)
(a FastAPI image-search service, a [pydantic-ai](https://ai.pydantic.dev)
agent) and the root [`README.md`](../README.md) for installation.
