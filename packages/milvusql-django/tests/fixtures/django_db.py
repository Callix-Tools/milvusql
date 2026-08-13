"""Shared Django/real-Milvus fixtures for the whole suite -- moved
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
def connection(milvus_server_target, milvus_db_name, _milvus_worker_cleanup):
    """A ``DatabaseWrapper`` pointed at the real Milvus server started
    by ``tests.fixtures.containers.milvus_server`` -- this worker's
    dedicated database, with every collection dropped after the test
    by ``_milvus_worker_cleanup`` (a dependency here, not this
    fixture's own teardown, so it runs *after* Django's own connection
    is closed below on the next test's setup). Closes the previous
    (possibly already-open) connection first so Django's per-thread
    connection cache doesn't hand back a stale handle."""
    connections["default"].close()
    host, port = milvus_server_target
    connections.databases["default"]["HOST"] = host
    connections.databases["default"]["PORT"] = port
    connections.databases["default"]["NAME"] = milvus_db_name
    connections.databases["default"]["USER"] = "root"
    connections.databases["default"]["PASSWORD"] = "Milvus"
    # 'Strong', not Milvus's own 'Bounded' default: against a real
    # (non-Lite) server, 'Bounded' permits exactly the staleness
    # window its name promises, so an ORM query issued immediately
    # after a create/update in the same test can legitimately not see
    # it yet -- confirmed directly against a real server (never
    # observable against Milvus Lite, which has no replication lag to
    # be inconsistent about). `base.py`'s `get_connection_params`
    # merges `OPTIONS` straight into `milvusql.dbapi.connect()`'s
    # kwargs, so this becomes the connection-level consistency
    # fallback every query without its own gets.
    connections.databases["default"]["OPTIONS"] = {
        "consistency_level": "Strong"
    }
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
