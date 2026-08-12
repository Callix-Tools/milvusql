"""Shared Django/Milvus Lite fixtures for the whole suite -- moved
verbatim from the old flat ``tests/conftest.py`` (everything below its
``settings.configure()``/``django.setup()`` bootstrap, which has to
stay top-level in ``tests/conftest.py`` itself; see that file's
docstring)."""

from __future__ import annotations

import typing as t

import pytest
from dj_app.models import Item as _Item
from django.db import connections
from milvusql_django.schema import create_index_and_load

#: No `django-stubs` in this workspace, so `ty` can't see the
#: metaclass-injected `Item._meta` -- every attribute this unlocks is
#: exercised for real, just invisible statically. Same kind of
#: understood cast ``test_translate.py`` uses for ``MilvusClient``.
Item: t.Any = _Item


@pytest.fixture
def connection(tmp_path):
    """A fresh ``DatabaseWrapper`` pointed at its own Milvus Lite
    file -- closes the previous (possibly already-open) connection
    first so Django's per-thread connection cache doesn't hand back a
    handle into a prior test's deleted file."""
    connections["default"].close()
    connections.databases["default"]["NAME"] = str(tmp_path / "milvus.db")
    return connections["default"]


@pytest.fixture
def loaded_items(connection):
    """The ``dj_app.models.Item`` collection, indexed and loaded --
    everything ``vector_search``/``hybrid_search``/``filter()`` need
    to run for real."""
    with connection.schema_editor() as editor:
        editor.create_model(Item)
    create_index_and_load(
        connection,
        Item._meta.db_table,
        "embedding",
        using="HNSW",
        metric_type="COSINE",
        M=16,
        ef_construction=200,
    )
    return connection
