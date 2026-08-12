# milvusql

A [PEP 249](https://peps.python.org/pep-0249/) DBAPI for [Milvus](https://milvus.io) — sync and
async, parsing/generating [MilvusQL](https://github.com/Callix-Tools/sqlglot-milvus) via
`sqlglot-milvus` and executing the resulting AST against `pymilvus`.

This is the core of a `uv` workspace: `milvusql-sqlalchemy` and `milvusql-django` (planned) depend
on this package as their DBAPI layer, each installed separately.

Status: early development, not yet published.

## License

MIT
