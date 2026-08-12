# milvusql-django

A Django database backend for [Milvus](https://milvus.io), built on the
[`milvusql`](https://github.com/Callix-Tools/milvusql) DBAPI.

```bash
pip install milvusql-django
```

```python
# settings.py
DATABASES = {
    "default": {
        "ENGINE": "milvusql_django",
        "NAME": "/path/to/items.db",  # or HOST/PORT/USER/PASSWORD for a real server
    }
}
```

`Model`/`Field` CRUD and filtering go through Django's normal compiler; vector and hybrid search
go through explicit helpers instead — see the docs for why.

**Full docs: https://Callix-Tools.github.io/milvusql-docs/docs/django/overview**

Status: early development, not yet published. The schema/migration layer in particular is a
first cut — see [Schema & Migrations](https://Callix-Tools.github.io/milvusql-docs/docs/django/schema-and-migrations)
for exactly what that does and doesn't cover.

## License

MIT
