# milvusql

A [PEP 249](https://peps.python.org/pep-0249/) DBAPI for [Milvus](https://milvus.io) — sync and
async — parsing/generating MilvusQL via [`sqlglot-milvus`](https://github.com/Callix-Tools/sqlglot-milvus)
and executing the resulting AST against `pymilvus`.

```bash
pip install milvusql
```

```python
import milvusql

conn = milvusql.connect(uri="./items.db")  # Milvus Lite, or a real server's URI
cur = conn.cursor()
cur.execute("SELECT id FROM items WHERE category = :cat LIMIT 10", {"cat": "book"})
print(cur.fetchall())
```

This is the core of a `uv` workspace: [`milvusql-sqlalchemy`](packages/milvusql-sqlalchemy) and
[`milvusql-django`](packages/milvusql-django) depend on this package as their DBAPI layer, each
installed separately.

**Full docs: https://Callix-Tools.github.io/milvusql-docs/**

Status: early development, not yet published.

## License

MIT
