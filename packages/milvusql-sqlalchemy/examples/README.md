# Examples

Small, runnable applications showing how to use `milvusql-sqlalchemy`
outside of the test suite. Each subdirectory is standalone: its own
`README.md` and `requirements.txt`, no shared code to chase across
directories.

| Example | Shows |
| --- | --- |
| [`fastapi_image_search/`](fastapi_image_search) | A FastAPI service for image search: insert images with a caption, search by a query image or by free text (cross-modal, via CLIP), delete. |
| [`pydantic_ai_agent/`](pydantic_ai_agent) | A [pydantic-ai](https://ai.pydantic.dev) shopping-assistant agent whose only access to the catalog is through typed tools backed by a real vector search — the model never sees SQL or the database directly. |

Both share the same production lesson worth reading even if you only need
one of them: schema bootstrap (`CREATE TABLE`/`CREATE INDEX`/`LOAD TABLE`)
runs through the *sync* dialect even in an otherwise fully-async app —
see either example's `db.py` docstring for why.

See also the root `milvusql` package's own examples in
[`examples/`](../../../examples) (a basic DBAPI walkthrough, a Temporal
ingestion worker) and this package's own [`README.md`](../README.md) for
installation.
