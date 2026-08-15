<div align="center">
  <img src="https://Callix-Tools.github.io/milvusql-docs/img/logo.svg" alt="milvusql logo" width="220"/>

  <h1>milvusql-django</h1>

  <p>A Django database backend for <a href="https://milvus.io">Milvus</a>, built on the <a href="https://github.com/Callix-Tools/milvusql"><code>milvusql</code></a> DBAPI.</p>

  [![PyPI](https://img.shields.io/pypi/v/milvusql-django?color=blue)](https://pypi.org/project/milvusql-django/)
  [![Python](https://img.shields.io/pypi/pyversions/milvusql-django)](https://pypi.org/project/milvusql-django/)
  [![License](https://img.shields.io/github/license/Callix-Tools/milvusql)](LICENSE)
  [![CI](https://img.shields.io/github/actions/workflow/status/Callix-Tools/milvusql/ci-django.yml?label=CI)](https://github.com/Callix-Tools/milvusql/actions)

  [📚 Documentation](https://Callix-Tools.github.io/milvusql-docs/docs/django/overview) · [PyPI](https://pypi.org/project/milvusql-django/) · [milvusql](https://github.com/Callix-Tools/milvusql)
</div>

---

`Model`/`Field` CRUD and filtering go through Django's normal query compiler; vector search goes through an explicit helper instead of a queryset method, because Milvus needs an index built and the collection loaded before a vector column is searchable at all — see [API](#api) below. Relation lookups (`filter(related__field=...)`), `.values().annotate(...)` grouping and `Exists(... OuterRef(...))` (correlated `EXISTS`, decorrelated into a semi/anti join) all plan through the DBAPI's relational engine, and a `TextField` is Milvus's analyzer-enabled full-text input.

## Installation

```bash
pip install milvusql-django
```

## Quick start

```python
# settings.py
DATABASES = {
    "default": {
        "ENGINE": "milvusql_django",
        "NAME": "/path/to/items.db",  # or HOST/PORT/USER/PASSWORD for a real server
    }
}
```

```python
# models.py
from django.db import models
from milvusql_django.fields import VectorField

class Item(models.Model):
    embedding = VectorField(dim=768)
    category = models.CharField(max_length=64)
```

```python
# after migrating: build the index and load the collection once
from django.db import connection
from milvusql_django.schema import create_index_and_load

create_index_and_load(
    connection, "myapp_item", "embedding",
    using="HNSW", metric_type="COSINE",
)
```

```python
# CRUD/filtering: the normal ORM
Item.objects.filter(category="book").values("id")

# vector search: a raw SQL escape hatch, not a queryset method
with connection.cursor() as cursor:
    cursor.execute(
        'SELECT id FROM "myapp_item" ORDER BY embedding <=> %s LIMIT 5',
        [[0.1] * 768],
    )
    rows = cursor.fetchall()
```

## API

### `VectorField`

A standard Django `Field`, not a fake-column shim — round-trips `list[float] <-> VECTOR(n)`:

```python
from milvusql_django.fields import VectorField

class Item(models.Model):
    embedding = VectorField(dim=768)   # dim is optional; omit for an unconstrained VECTOR
```

### `milvusql_django.schema.create_index_and_load()`

```python
create_index_and_load(
    connection,          # a Django database connection
    table,                # collection/table name
    field_name,            # the VectorField's column
    *,
    using="HNSW",
    metric_type="COSINE",
    **index_params,        # e.g. M=16, ef_construction=200
)
```

The explicit follow-up step `CreateModel` deliberately doesn't do automatically — index method/metric is a query-shape decision (HNSW vs. IVF, COSINE vs. L2), not something a generic schema migration should guess. Call it once, after defining the model, before querying it.

## Schema & migrations

This is a first cut, not full Django migration parity:

| Operation | Support |
|---|---|
| `CreateModel` (scalar fields + one or more `VectorField`s) | ✅ |
| `AddField` | ✅ against a real Milvus server — Milvus Lite's gRPC server doesn't implement `AddCollectionField` |
| `RemoveField`, `AlterField` | ❌ raises loudly — Milvus can't alter or drop a field |

See [Schema & Migrations](https://Callix-Tools.github.io/milvusql-docs/docs/django/schema-and-migrations) for the full picture.

## Development

From the workspace root (requires Python 3.12+, [uv](https://docs.astral.sh/uv/), [task](https://taskfile.dev/)):

```bash
task install
task django:lint
task django:test    # integration tests need Docker (testcontainers)
```

## License

MIT
