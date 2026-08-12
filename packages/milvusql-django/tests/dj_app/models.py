"""A minimal real, registered model -- ``VectorField``/model-level
integration tests need an actual app in ``INSTALLED_APPS`` (Django's
model metaclass requires ``django.setup()`` to have populated the app
registry first; ``isolate_apps()`` doesn't work for a throwaway label
with no backing package)."""

from __future__ import annotations

from django.db import models
from milvusql_django.fields import VectorField


class Item(models.Model):
    category = models.CharField(max_length=64)
    rank = models.IntegerField()
    embedding = VectorField(dim=8)

    class Meta:
        app_label = "dj_app"
