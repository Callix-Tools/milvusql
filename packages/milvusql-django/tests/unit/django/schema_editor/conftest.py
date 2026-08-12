"""Shared fixtures for the ``schema_editor`` action's ``act.py``/
``error.py``: a bare ``DatabaseWrapper`` and a minimal duck-typed
``Item`` stand-in, built the same way for both the happy-path DDL
tests and the unsupported-alteration error tests."""

from __future__ import annotations

import pytest
from django.db import models
from milvusql_django.base import DatabaseWrapper
from milvusql_django.fields import VectorField


def _named(field, name):
    field.set_attributes_from_name(name)
    return field


@pytest.fixture
def connection():
    settings_dict = {
        "NAME": "mydb",
        "OPTIONS": {},
        "TIME_ZONE": None,
        "USER": None,
        "PASSWORD": None,
        "HOST": None,
        "PORT": None,
        "AUTOCOMMIT": True,
        "CONN_MAX_AGE": 0,
        "CONN_HEALTH_CHECKS": False,
    }
    return DatabaseWrapper(settings_dict, alias="default")


@pytest.fixture
def item_model():
    pk = _named(models.BigAutoField(primary_key=True), "id")
    category = _named(models.CharField(max_length=64), "category")
    embedding = _named(VectorField(dim=8), "embedding")

    class Meta:
        db_table = "items"
        local_fields = [pk, category, embedding]
        auto_field = pk

    return type("Item", (), {"_meta": Meta})
