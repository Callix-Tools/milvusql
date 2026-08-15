<div align="center">
  <img src="https://Callix-Tools.github.io/milvusql-docs/img/logo.svg" alt="milvusql logo" width="220"/>

  <h1>milvusql-sqlalchemy</h1>

  <p>A SQLAlchemy 2.0 dialect for <a href="https://milvus.io">Milvus</a>, built on the <a href="https://github.com/Callix-Tools/milvusql"><code>milvusql</code></a> DBAPI.</p>

  [![PyPI](https://img.shields.io/pypi/v/milvusql-sqlalchemy?color=blue)](https://pypi.org/project/milvusql-sqlalchemy/)
  [![Python](https://img.shields.io/pypi/pyversions/milvusql-sqlalchemy)](https://pypi.org/project/milvusql-sqlalchemy/)
  [![License](https://img.shields.io/github/license/Callix-Tools/milvusql)](LICENSE)
  [![CI](https://img.shields.io/github/actions/workflow/status/Callix-Tools/milvusql/ci-sqlalchemy.yml?label=CI)](https://github.com/Callix-Tools/milvusql/actions)

  [📚 Documentation](https://Callix-Tools.github.io/milvusql-docs/docs/sqlalchemy/overview) · [PyPI](https://pypi.org/project/milvusql-sqlalchemy/) · [milvusql](https://github.com/Callix-Tools/milvusql)
</div>

---

Core/ORM `select()`, DDL (`CREATE TABLE`/`CREATE INDEX`), reflection, and Alembic migrations all go through the standard SQLAlchemy surface — vector search is just `.order_by(column.cosine_distance(q))`, spelled exactly like [pgvector](https://github.com/pgvector/pgvector-python)'s own comparator.

## Installation

```bash
pip install milvusql-sqlalchemy
```

## Quick start

```python
from sqlalchemy import create_engine, select
from milvusql_sqlalchemy.types import VECTOR

engine = create_engine("milvusql:///items.db")  # Milvus Lite, or a real server's URI

with engine.connect() as conn:  # items: Table with an `embedding` VECTOR column
    rows = conn.execute(
        select(items.c.id, items.c.category)
        .order_by(items.c.embedding.l2_distance([0.1] * 768))
        .limit(5)
    ).all()
```

The same engine, asyncio-native, via `create_async_engine` and the `milvusql+aio` driver:

```python
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine("milvusql+aio:///items.db")

async with engine.connect() as conn:
    rows = (
        await conn.execute(
            select(items.c.id).order_by(items.c.embedding.cosine_distance(q)).limit(5)
        )
    ).all()
```

## Connection URLs

```
milvusql:///items.db                              # Milvus Lite -- relative file path
milvusql:////abs/path/items.db                     # Milvus Lite -- absolute file path
milvusql://user:password@host:19530/db_name        # a real server
milvusql://user:password@host:19530/db_name?secure=1  # TLS (https)
milvusql+aio://...                                 # same URL forms, async engine
```

## API

### `VECTOR(dim)` / `SPARSEVEC`

```python
from sqlalchemy import Column, Table
from milvusql_sqlalchemy.types import VECTOR, SPARSEVEC

items = Table(
    "items", metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("embedding", VECTOR(768)),
    Column("sparse", SPARSEVEC),
    milvusql_shards=1,
    milvusql_consistency_level="Strong",
)
```

| Comparator method | MilvusQL operator | Available on |
|---|:---:|---|
| `.l2_distance(other)` | `<->` | `VECTOR` |
| `.cosine_distance(other)` | `<=>` | `VECTOR` |
| `.l1_distance(other)` | `<+>` | `VECTOR` |
| `.max_inner_product(other)` | `<#>` | `VECTOR`, `SPARSEVEC` |

Bind values pass through as plain `list[float]` — never stringified into SQL text.

### `hybrid_search()`

Weighted multi-vector search, rendered where MilvusQL's `HYBRID SEARCH ... RERANK ...` sits — in place of `ORDER BY`:

```python
from milvusql_sqlalchemy.hybrid import hybrid_search, weighted

select(items.c.id).order_by(
    hybrid_search(
        weighted(items.c.embedding.cosine_distance(dense_q), 0.7),
        weighted(items.c.sparse.max_inner_product(sparse_q), 0.3),
        rerank="RRF", k=60,
    )
).limit(10)
```

### DDL options

Dialect-specific `Table`/`Index` arguments, the same mechanism as `mysql_engine=...`:

```python
Table(
    "items", metadata, ...,
    milvusql_shards=2,
    milvusql_consistency_level="Bounded",
    milvusql_partition_key="category",
)
Index(
    "idx_embedding", items.c.embedding,
    milvusql_using="HNSW",
    milvusql_with={"metric_type": "COSINE", "M": 16, "ef_construction": 200},
)
```

### Consistency level

A statement's own `CONSISTENCY LEVEL` clause always wins; otherwise the engine falls back to the connection's isolation level — one of Milvus's own levels (`Strong`, `Bounded`, `Eventually`, `Session`, `Customized`), set the normal SQLAlchemy way:

```python
conn = engine.connect().execution_options(isolation_level="Bounded")
```

### Alembic

Importing this dialect registers `milvusql`'s `DefaultImpl` with Alembic automatically — `alembic upgrade`/`downgrade` work against a `milvusql://...` URL with no extra setup. Non-transactional (`transactional_ddl = False`): Milvus has no multi-statement rollback, so a failed migration can't be undone by Alembic — write forward-only migrations.

## Examples

| Example | Shows |
|---|---|
| [`examples/fastapi_image_search`](examples/fastapi_image_search) | A FastAPI service for image search: insert images with a caption, search by query image or free text (cross-modal, via CLIP), delete |
| [`examples/pydantic_ai_agent`](examples/pydantic_ai_agent) | A [pydantic-ai](https://ai.pydantic.dev) shopping-assistant agent whose only access to the catalog is through typed tools backed by a real vector search |

## Development

From the workspace root (requires Python 3.12+, [uv](https://docs.astral.sh/uv/), [task](https://taskfile.dev/)):

```bash
task install
task sqlalchemy:lint
task sqlalchemy:test    # integration tests need Docker (testcontainers)
```

## License

MIT
