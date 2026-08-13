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
* otherwise -> a ``testcontainers`` ``MilvusContainer`` is started --
  exactly ONE for the whole test session, not one per pytest-xdist
  worker: Milvus standalone's own recommended minimum is 4 CPU/8GB
  RAM, so one container per worker (``-n auto``/``CONCURRENCY>1``)
  starts several full Milvus instances competing for the same CI
  runner's resources at once, which reliably OOM-aborts (confirmed
  directly: ``exit code 134``/SIGABRT under `CONCURRENCY=4`). Only the
  first worker to grab a ``filelock`` actually starts the container;
  every other worker reads its host/port back from a small JSON file
  in the base tmp dir all workers share (the standard pytest-xdist
  "run a session fixture exactly once" recipe) -- see
  ``milvus_container``/``_container_info_file``.

The one container (or env-provided server) is shared by every worker.
Isolation across workers is a Milvus *database* named after the worker
id (``test_gw0``, ``test_gw1``, ..., ``test_master`` when not running
under ``-n``); isolation across tests *within* one worker's database
is a per-test collection-drop teardown, since Milvus Lite's old "fresh
on-disk file per test" trick no longer applies to a real, shared
server.

Either way, the container is only ever started when a collected test
actually carries ``@pytest.mark.integration`` -- a plain
``-m unit`` run never touches Docker or the network.
"""

from __future__ import annotations

import json
import os
from urllib.parse import urlsplit

import pytest
from filelock import FileLock
from pymilvus import MilvusClient
from testcontainers.community.milvus import MilvusContainer
from testcontainers.core.wait_strategies import (
    CompositeWaitStrategy,
    HttpWaitStrategy,
    LogMessageWaitStrategy,
)

# Default super-user credential documented by pymilvus itself for a
# freshly started standalone Milvus server (root user, password
# "Milvus") -- not a secret, just the well-known out-of-the-box login.
_ROOT_TOKEN = "root:Milvus"  # nosec B105 -- well-known default, not a secret

_IMAGE = "milvusdb/milvus:v2.6.22"


def _has_integration_marker(request: pytest.FixtureRequest) -> bool:
    return any(
        item.get_closest_marker("integration") is not None
        for item in request.session.items
    )


def _start_milvus_container() -> MilvusContainer:
    """Starts one ``MilvusContainer``, matching Milvus's own official
    single-container startup script (``scripts/standalone_embed.sh``).

    Three real divergences from that script/from a normal dev machine,
    each confirmed directly against this exact image/environment
    combination via successive real CI failures:

    1. ``testcontainers``' ``MilvusContainer`` sets ``ETCD_USE_EMBED``
       but never ``DEPLOY_MODE=STANDALONE`` -- Milvus defaults to
       distributed mode otherwise, and distributed mode + embedded
       etcd is an explicit, hard panic in Milvus's own startup code:
       ``panic: embedded etcd can not be used under distributed
       mode`` (``EtcdConfig.Init``, confirmed verbatim from a real CI
       failure's captured container stderr). Without
       ``DEPLOY_MODE=STANDALONE``, this container can never start.
    2. It also never passes ``--security-opt seccomp:unconfined``,
       which the official script does -- a restrictive default
       seccomp profile (as some CI runners use) can make Milvus's own
       process abort during startup with no useful message.
    3. ``MilvusContainer``'s own default wait-strategy timeout
       (``testcontainers_config.timeout``, 120s) is too short on a
       CPU-constrained CI runner -- confirmed directly: a real CI run
       took well over 120s to emit ``"Welcome to use Milvus!"`` even
       after fixes 1-2 let the process actually start, timing out
       repeatedly. Milvus's own compose healthcheck already budgets a
       90s ``start_period``; 300s here is a deliberately generous
       margin over that on top of a slower, shared CI CPU.

    On any startup failure, re-raise with the container's own
    stdout/stderr attached -- the bare ``RuntimeError`` a failed wait
    strategy raises otherwise carries an empty ``Container error: .``,
    which is not enough to diagnose a real crash from."""
    container = (
        MilvusContainer(image=_IMAGE)
        .with_env("DEPLOY_MODE", "STANDALONE")
        .with_kwargs(security_opt=["seccomp:unconfined"])
    )
    container.waiting_for(
        CompositeWaitStrategy(
            LogMessageWaitStrategy("Welcome to use Milvus!"),
            HttpWaitStrategy(container.healthcheck_port, "/healthz"),
        ).with_startup_timeout(300)
    )
    try:
        container.start()
    except Exception as exc:
        try:
            stdout, stderr = container.get_logs()
        except Exception:
            stdout, stderr = b"", b""
        msg = (
            f"Milvus container failed to start: {exc}\n"
            f"--- container stdout ---\n{stdout.decode(errors='replace')}\n"
            f"--- container stderr ---\n{stderr.decode(errors='replace')}"
        )
        raise RuntimeError(msg) from exc
    return container


def _container_info_file(tmp_path_factory: pytest.TempPathFactory):
    """Path to the small JSON file every pytest-xdist worker shares
    (``tmp_path_factory.getbasetemp().parent`` is the base tmp dir
    common to all workers, unlike ``getbasetemp()`` itself, which is
    already per-worker) -- whoever writes ``{"host":..., "port":...}``
    here first is the one worker that actually started the container.
    A plain function, not a fixture: it must be safe to call from
    inside an already-gated branch (``MILVUS_TEST_URI`` set, or no
    ``integration`` test collected) without ever touching Docker --
    declaring it as a fixture dependency instead would make pytest
    resolve it unconditionally, defeating that gating entirely."""
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    return root_tmp_dir / "milvus_container_target.json"


@pytest.fixture(scope="session")
def milvus_container(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
):
    """Starts the ONE Milvus container for this whole test session --
    not one per pytest-xdist worker: Milvus standalone's own
    recommended minimum is 4 CPU/8GB RAM, so one container per worker
    (``-n auto``/``CONCURRENCY>1``) starts several full Milvus
    instances competing for the same CI runner's resources at once,
    which reliably OOM-aborts (confirmed directly: ``exit code
    134``/SIGABRT under `CONCURRENCY=4`). Only the first worker to grab
    a ``filelock`` on ``_container_info_file`` actually starts it and
    writes its host/port there (the standard pytest-xdist "run a
    session fixture exactly once" recipe); every other worker just
    reads that file back via ``milvus_uri`` below, without holding the
    lock while waiting on the container's own ~90s startup.

    Yields ``None`` when no ``integration`` test was collected, or
    when ``MILVUS_TEST_URI`` already points at a server to use
    instead -- in both cases nothing here ever touches Docker or
    ``tmp_path_factory``'s shared file."""
    if not _has_integration_marker(request):
        yield None
        return
    if os.getenv("MILVUS_TEST_URI"):
        yield None  # env DSN wins, no container needed
        return

    info_file = _container_info_file(tmp_path_factory)
    owned_container = None
    with FileLock(str(info_file) + ".lock"):
        if not info_file.is_file():
            owned_container = _start_milvus_container()
            info_file.write_text(
                json.dumps(
                    {
                        "host": owned_container.get_container_host_ip(),
                        "port": int(owned_container.get_exposed_port(19530)),
                    }
                )
            )
    try:
        yield owned_container
    finally:
        if owned_container:
            owned_container.stop()


@pytest.fixture(scope="session")
def milvus_uri(
    milvus_container, tmp_path_factory: pytest.TempPathFactory
) -> str:
    """Resolves to the ``http://host:port`` URI of the real Milvus
    server to test against, regardless of whether it came from
    ``MILVUS_TEST_URI`` or the one shared container. Depends on
    ``milvus_container`` (even though it doesn't use its value) purely
    for fixture-resolution ordering -- the container must actually be
    started before any URI pointing at it is handed out; by then,
    ``_container_info_file`` is guaranteed to exist, whether this
    worker wrote it or another one did."""
    if uri := os.getenv("MILVUS_TEST_URI"):
        parts = urlsplit(uri)
        if parts.hostname is None:
            msg = f"MILVUS_TEST_URI is missing a host: {uri!r}"
            raise ValueError(msg)
        return uri
    data = json.loads(_container_info_file(tmp_path_factory).read_text())
    return f"http://{data['host']}:{data['port']}"


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
