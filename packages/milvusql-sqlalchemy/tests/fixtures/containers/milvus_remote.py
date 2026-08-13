"""Fixtures for ``validation``-marked tests that need a real,
non-Lite Milvus server -- exercising the ``milvusql://`` /
``milvusql+aio://`` URL shape that carries a real host, port, and
``user:password`` credential pair through the dialect's
``create_connect_args`` (the file/no-host branch is already covered
by ``tests/fixtures/engine.py``'s Milvus-Lite fixtures; this module
covers the branch those never reach).

Infra selection lives here, in the fixture, never as a branch inside
a test body:

* ``MILVUS_VALIDATION_URI`` set -> that server is used as-is, no
  container is started.
* otherwise -> a ``testcontainers`` ``MilvusContainer`` is started,
  scoped to the pytest session (so, under ``pytest-xdist``, each
  worker gets its own container -- ``MilvusContainer`` binds to a
  random free host port, so no manual per-worker port/schema
  bookkeeping is needed the way it would be for e.g. Postgres).

Either way, the container is only ever started when a collected test
actually carries ``@pytest.mark.validation`` -- a plain
``-m "not validation"`` run (the default, see the taskfile) never
touches Docker or the network.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import pytest
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

# Default super-user credential documented by pymilvus itself for a
# freshly started standalone Milvus server (root user, password
# "Milvus") -- not a secret, just the well-known out-of-the-box login.
_ROOT_USER = "root"
_ROOT_PASSWORD = "Milvus"


def _has_validation_marker(request: pytest.FixtureRequest) -> bool:
    return any(
        item.get_closest_marker("validation") is not None
        for item in request.session.items
    )


@pytest.fixture(scope="session")
def milvus_container(request: pytest.FixtureRequest):
    """Starts a real Milvus server via ``testcontainers`` -- but only
    when the collected run actually needs one. Yields ``None`` when no
    ``validation`` test was collected, or when ``MILVUS_VALIDATION_URI``
    already points at a server to use instead: in both cases
    ``milvus_remote_target`` below is what tests/fixtures actually
    consult."""
    if not _has_validation_marker(request):
        yield None
        return
    if os.getenv("MILVUS_VALIDATION_URI"):
        yield None  # env DSN wins, no container needed
        return

    from testcontainers.community.milvus import MilvusContainer

    with MilvusContainer(image="milvusdb/milvus:v2.6.22") as container:
        yield container


@pytest.fixture(scope="session")
def milvus_remote_target(milvus_container) -> tuple[str, int]:
    """Resolves to the ``(host, port)`` of the real Milvus server to
    test against, regardless of whether it came from
    ``MILVUS_VALIDATION_URI`` or a freshly started container."""
    if uri := os.getenv("MILVUS_VALIDATION_URI"):
        parts = urlsplit(uri)
        return parts.hostname, parts.port or 19530
    host = milvus_container.get_container_host_ip()
    port = int(milvus_container.get_exposed_port(19530))
    return host, port


@pytest.fixture
def remote_engine(milvus_remote_target):
    """A sync ``Engine`` built from a real
    ``milvusql://user:pass@host:port/db`` URL string -- proving
    credentials, host, and port all round-trip through
    ``create_connect_args`` to a real, non-Lite server."""
    host, port = milvus_remote_target
    uri = f"milvusql://{_ROOT_USER}:{_ROOT_PASSWORD}@{host}:{port}/default"
    eng = create_engine(uri)
    yield eng
    eng.dispose()


@pytest.fixture
async def remote_engine_aio(milvus_remote_target):
    """The async equivalent of ``remote_engine``, built from a real
    ``milvusql+aio://user:pass@host:port/db`` URL string."""
    host, port = milvus_remote_target
    uri = f"milvusql+aio://{_ROOT_USER}:{_ROOT_PASSWORD}@{host}:{port}/default"
    eng = create_async_engine(uri)
    yield eng
    await eng.dispose()
