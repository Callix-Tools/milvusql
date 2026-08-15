# Examples

Small, runnable applications showing how to use `milvusql-sqlalchemy`
outside of the test suite. Each subdirectory is standalone: its own
`README.md`, `pyproject.toml`, and `docker-compose.yaml`, no shared
application code to chase across directories. Both target a real Milvus
server (via the bundled Compose file, not Milvus Lite) and are deliberately
**not** members of the repo's root `uv` workspace -- see either example's
`pyproject.toml` for why.

| Example | Shows |
| --- | --- |
| [`fastapi_image_search/`](fastapi_image_search) | A FastAPI service for image search: insert images with a caption, search by a query image or by free text (cross-modal, via CLIP), delete. |
| [`pydantic_ai_agent/`](pydantic_ai_agent) | A [pydantic-ai](https://ai.pydantic.dev) shopping-assistant agent whose only access to the catalog is through typed tools backed by a real vector search — the model never sees SQL or the database directly. |

Both share the same lesson worth reading even if you only need one of
them: against a real server, schema bootstrap (`CREATE TABLE`/`CREATE
INDEX`) can run entirely through the async dialect, with no explicit `LOAD
TABLE` step either -- see either example's `db.py` docstring for why (the
short version: both of those are Milvus-Lite-specific gaps this package's
async dialect no longer has to work around here).

See also the root `milvusql` package's own examples in
[`examples/`](../../../examples) (a basic DBAPI walkthrough, a Temporal
ingestion worker) and this package's own [`README.md`](../README.md) for
installation.
