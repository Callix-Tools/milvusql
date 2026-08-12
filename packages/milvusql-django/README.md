# milvusql-django

A Django database backend for [Milvus](https://milvus.io), built on the
[`milvusql`](https://github.com/Callix-Tools/milvusql) DBAPI.

```python
DATABASES = {
    "default": {
        "ENGINE": "milvusql_django",
        "NAME": "/path/to/milvus_lite.db",  # or HOST/PORT for a real server
    }
}
```

Standard `Model`/`Field` CRUD and scalar filtering work through Django's normal query
compiler. Vector and hybrid search have no relational equivalent, so they go through
explicit helpers instead of `.filter()`:

```python
from milvusql_django.fields import VectorField
from milvusql_django.expressions import vector_search

class Item(models.Model):
    category = models.CharField(max_length=64)
    embedding = VectorField(dim=768)

results = vector_search(Item, "embedding", query_vector, k=5, category="book")
```

Status: early development, not yet published. See the design plan's D11 for what is and
isn't covered yet -- the schema/migration layer in particular is a first cut, not a
finished implementation.

## License

MIT
