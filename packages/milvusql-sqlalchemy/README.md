# milvusql-sqlalchemy

A SQLAlchemy 2.0 dialect for [Milvus](https://milvus.io), built on the
[`milvusql`](https://github.com/Callix-Tools/milvusql) DBAPI (which in turn
uses [`sqlglot-milvus`](https://github.com/Callix-Tools/sqlglot-milvus) as its
parser).

```python
from sqlalchemy import create_engine, select, MetaData, Table, Column, Integer, String
from milvusql_sqlalchemy.types import VECTOR

engine = create_engine("milvusql://localhost:19530/default")

metadata = MetaData()
items = Table(
    "items", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("category", String(64)),
    Column("embedding", VECTOR(768)),
)
metadata.create_all(engine)

with engine.connect() as conn:
    rows = conn.execute(
        select(items.c.id, items.c.category)
        .order_by(items.c.embedding.l2_distance([0.1] * 768))
        .limit(5)
    ).all()
```

Status: early development, not yet published.

## License

MIT
