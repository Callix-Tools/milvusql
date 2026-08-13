"""Fixtures backing ``integration``-marked tests with a REAL Milvus
server -- ``integration`` now means "exercises a real Milvus server",
not "opt-in tier" (that distinction, ``validation``, has been retired
repo-wide).

Infra selection lives here, in the fixture, never as a branch inside a
test body:

* ``MILVUS_TEST_URI`` set -> that server is used as-is, no container
  is started. Expected shape: a full ``http://host:port`` URI (same
  shape ``milvusql.connect(uri=...)``/``MilvusClient(uri=...)``
  accept).
* otherwise -> a ``testcontainers`` ``MilvusContainer`` is started,
  scoped to the pytest session (so, under ``pytest-xdist``, each
  worker gets its own container -- ``MilvusContainer`` binds to a
  random free host port, so no manual per-worker port bookkeeping is
  needed the way it would be for e.g. Postgres).

The container (or env-provided server) is shared by every test in one
worker process. Isolation across workers is a Milvus *database* named
after the worker id (``test_gw0``, ``test_gw1``, ..., ``test_master``
when not running under ``-n``); isolation across tests *within* one
worker's database is a per-test collection-drop teardown, since
Milvus Lite's old "fresh on-disk file per test" trick no longer
applies to a real, shared server.

Either way, the container is only ever started when a collected test
actually carries ``@pytest.mark.integration`` -- a plain
``-m unit`` run never touches Docker or the network.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import pytest
from pymilvus import MilvusClient
from testcontainers.community.milvus import MilvusContainer

# Default super-user credential documented by pymilvus itself for a
# freshly started standalone Milvus server (root user, password
# "Milvus") -- not a secret, just the well-known out-of-the-box login.
_ROOT_TOKEN = "root:Milvus"  # nosec B105 -- well-known default, not a secret


def _has_integration_marker(request: pytest.FixtureRequest) -> bool:
    return any(
        item.get_closest_marker("integration") is not None
        for item in request.session.items
    )


@pytest.fixture(scope="session")
def milvus_container(request: pytest.FixtureRequest):
    """Starts a real Milvus server via ``testcontainers`` -- but only
    when the collected run actually needs one. Yields ``None`` when no
    ``integration`` test was collected, or when ``MILVUS_TEST_URI``
    already points at a server to use instead: in both cases
    ``milvus_uri`` below is what tests/fixtures actually consult."""
    if not _has_integration_marker(request):
        yield None
        return
    if os.getenv("MILVUS_TEST_URI"):
        yield None  # env DSN wins, no container needed
        return

    with MilvusContainer(image="milvusdb/milvus:v2.6.22") as container:
        yield container


@pytest.fixture(scope="session")
def milvus_uri(milvus_container) -> str:
    """Resolves to the ``http://host:port`` URI of the real Milvus
    server to test against, regardless of whether it came from
    ``MILVUS_TEST_URI`` or a freshly started container."""
    if uri := os.getenv("MILVUS_TEST_URI"):
        parts = urlsplit(uri)
        if parts.hostname is None:
            msg = f"MILVUS_TEST_URI is missing a host: {uri!r}"
            raise ValueError(msg)
        return uri
    host = milvus_container.get_container_host_ip()
    port = milvus_container.get_exposed_port(19530)
    return f"http://{host}:{port}"


@pytest.fixture(scope="session")
def milvus_db_name(milvus_uri: str, worker_id: str) -> str:
    """A Milvus database dedicated to this pytest-xdist worker
    (``worker_id`` is ``"master"`` outside ``-n``), so parallel
    workers never see each other's collections. Created once per
    worker via a plain, throwaway control-plane client -- not through
    the ``milvusql`` DBAPI, since this is test setup, not the code
    under test."""
    name = f"test_{worker_id}"
    client = MilvusClient(uri=milvus_uri, token=_ROOT_TOKEN)
    try:
        if name not in client.list_databases():
            client.create_database(name)
    finally:
        client.close()
    return name


@pytest.fixture
def _milvus_worker_cleanup(milvus_db_name: str, milvus_uri: str):
    """Drops every collection left behind in this worker's database
    after each test. All tests in a worker share one database, so
    this replaces Milvus Lite's old "fresh file per test" isolation --
    wired as a dependency of ``conn``/``aconn``, never as a bare
    ``autouse=True``, so it never fires for ``unit`` tests."""
    yield
    client = MilvusClient(uri=milvus_uri, token=_ROOT_TOKEN, db_name=milvus_db_name)
    try:
        for collection in client.list_collections():
            client.drop_collection(collection)
    finally:
        client.close()
