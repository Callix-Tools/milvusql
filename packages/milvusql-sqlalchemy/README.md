# milvusql-sqlalchemy

A SQLAlchemy 2.0 dialect for [Milvus](https://milvus.io), built on the
[`milvusql`](https://github.com/Callix-Tools/milvusql) DBAPI.

```bash
pip install milvusql-sqlalchemy
```

```python
from sqlalchemy import create_engine, select
from milvusql_sqlalchemy.types import VECTOR

engine = create_engine(
    "milvusql:///items.db"
)  # Milvus Lite, or a real server's URI

with (
    engine.connect() as conn
):  # items: Table with an `embedding` VECTOR column
    rows = conn.execute(
        select(items.c.id, items.c.category)
        .order_by(items.c.embedding.l2_distance([0.1] * 768))
        .limit(5)
    ).all()
```

**Examples:** a FastAPI image-search service and a pydantic-ai agent live in
[`examples/`](examples).

**Full docs: https://Callix-Tools.github.io/milvusql-docs/docs/sqlalchemy/overview**

Status: early development, not yet published.

## License

MIT
