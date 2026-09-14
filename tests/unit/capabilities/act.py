"""Unit coverage for ``milvusql.capabilities`` -- the table that says
which side of the wire evaluates what, derived from the server's
reported version and the installed ``pymilvus``.

Pure logic, so every case here is a version string in and a flag set
out. The one non-obvious input is Milvus Lite's ``"milvus_lite-3.2.0"``
(taken verbatim from a real embedded connection, not invented): its
trailing number is the ``milvus-lite`` package version, and reading it
as a Milvus 3.2 server would switch on exactly the server-side flags
the embedded engine does not have.

The conservative direction is asserted deliberately and in both
directions -- an unknown or unreadable version must produce the
client-side answer this release already gives, never an optimistic one.
"""

from __future__ import annotations

import pytest
from pymilvus.exceptions import MilvusException

import milvusql
from milvusql import aio
from milvusql.capabilities import (
    ROW_CEILING_2X,
    capabilities_for,
    installed_client_release,
    parse_release,
)
from milvusql.dbapi import connection as connection_module

pytestmark = [pytest.mark.unit, pytest.mark.capabilities]

CLIENT_2_6 = (2, 6)
CLIENT_3_0 = (3, 0)


class TestParseRelease:
    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("v2.6.17", (2, 6)),
            ("2.6.17", (2, 6)),
            ("v2.6.4-gpu", (2, 6)),  # a real build tag carries suffixes
            ("v3.0.0", (3, 0)),
            ("v3.10.1", (3, 10)),  # minor is a number, not a digit
        ],
    )
    def test_reads_major_and_minor_off_a_build_tag(self, version, expected):
        assert parse_release(version) == expected

    @pytest.mark.parametrize("version", [None, "", "unknown", "milvus"])
    def test_unreadable_version_parses_to_none_not_a_guess(self, version):
        assert parse_release(version) is None

    def test_milvus_lite_package_version_is_not_a_server_release(self):
        """The whole reason Lite is special-cased: "3.2.0" here is the
        ``milvus-lite`` package, and treating it as Milvus 3.2 would
        claim kernel GROUP BY from the embedded engine."""
        assert parse_release("milvus_lite-3.2.0") is None

    def test_installed_client_release_is_the_pymilvus_in_use(self):
        release = installed_client_release()
        assert release is not None
        assert release >= CLIENT_2_6  # pyproject floors pymilvus at 2.6


class TestSupportedServer:
    """Milvus 2.6 through a 2.6 client -- the pairing this release
    actually supports, and the one whose flags must match what the code
    does today."""

    @pytest.fixture
    def caps(self):
        return capabilities_for("v2.6.4", client_release=CLIENT_2_6)

    def test_grouping_search_is_available(self, caps):
        """``group_by_field``/``group_size`` reach the wire from
        pymilvus 2.6 already (its ``prepare.py`` folds them into
        ``search_params``), so top-k-per-group does not wait for 3.0."""
        assert caps.grouping_search is True

    def test_relational_work_stays_client_side(self, caps):
        assert caps.server_side_group_by is False
        assert caps.server_side_order_by is False

    def test_text_is_not_a_native_field_yet(self, caps):
        assert caps.native_text_field is False

    def test_no_server_issued_cursor(self, caps):
        assert caps.search_iterator_cursor is False
        assert caps.query_iterator_cursor is False

    def test_row_ceiling_and_ordered_pages_match_todays_behaviour(self, caps):
        assert caps.per_call_row_ceiling == ROW_CEILING_2X
        assert caps.ordered_iterator_pages is True

    def test_it_reports_what_it_was_told(self, caps):
        assert caps.server_version == "v2.6.4"
        assert caps.server_release == (2, 6)
        assert caps.client_release == CLIENT_2_6
        assert caps.embedded is False


class TestKernelServer:
    """Milvus 3.0 reached by a client new enough to ask it for the
    work -- every flag the discussion pinned to the engine."""

    @pytest.fixture
    def caps(self):
        return capabilities_for("v3.0.1", client_release=CLIENT_3_0)

    def test_single_collection_sql_moves_into_the_kernel(self, caps):
        assert caps.server_side_group_by is True
        assert caps.server_side_order_by is True

    def test_text_and_cursors_are_native(self, caps):
        assert caps.native_text_field is True
        assert caps.search_iterator_cursor is True
        assert caps.query_iterator_cursor is True

    def test_the_fixed_row_ceiling_is_gone(self, caps):
        assert caps.per_call_row_ceiling is None


class TestClientGating:
    """A capable server is not enough: the client has to be able to
    spell the request."""

    def test_old_client_cannot_ask_a_new_server_for_kernel_group_by(self):
        """``group_by_fields`` appears nowhere in pymilvus 2.6.17
        (checked against the installed package), so this pairing has to
        keep planning the client-side path."""
        caps = capabilities_for("v3.0.1", client_release=CLIENT_2_6)
        assert caps.server_side_group_by is False
        assert caps.server_side_order_by is False
        assert caps.native_text_field is False
        assert caps.query_iterator_cursor is False

    def test_a_2_6_client_still_takes_a_3_0_search_cursor(self):
        """``SearchIteratorV2`` ships in pymilvus 2.6.17 -- this is the
        one 3.0 capability the current client can already consume, and
        the reason the search and query cursors are separate flags."""
        caps = capabilities_for("v3.0.1", client_release=CLIENT_2_6)
        assert caps.search_iterator_cursor is True

    def test_new_client_against_an_old_server_delegates_nothing(self):
        caps = capabilities_for("v2.6.4", client_release=CLIENT_3_0)
        assert caps.server_side_group_by is False
        assert caps.search_iterator_cursor is False
        assert caps.per_call_row_ceiling == ROW_CEILING_2X


class TestMilvusLite:
    @pytest.fixture
    def caps(self):
        return capabilities_for("milvus_lite-3.2.0", client_release=CLIENT_2_6)

    def test_it_is_recognised_as_embedded(self, caps):
        assert caps.embedded is True
        assert caps.server_release is None

    def test_nothing_is_delegated_to_the_embedded_engine(self, caps):
        assert caps.grouping_search is False
        assert caps.server_side_group_by is False
        assert caps.native_text_field is False
        assert caps.search_iterator_cursor is False

    def test_pages_are_not_assumed_ordered(self, caps):
        """Lite ignores the ``iterator`` flag and returns rows in
        arbitrary order -- the page loop in ``translate._common``
        raises on exactly this, and the flag agrees with it."""
        assert caps.ordered_iterator_pages is False

    def test_the_row_ceiling_still_applies(self, caps):
        assert caps.per_call_row_ceiling == ROW_CEILING_2X


class TestUnknownServer:
    """Nothing readable came back. Every flag has to fall to the
    client-side answer, because a wrong optimistic flag is a query that
    returns the wrong rows without saying so."""

    @pytest.fixture(params=[None, "", "some-fork-of-milvus"])
    def caps(self, request):
        return capabilities_for(request.param, client_release=CLIENT_3_0)

    def test_no_capability_is_assumed(self, caps):
        assert caps.server_release is None
        assert caps.embedded is False
        assert caps.grouping_search is False
        assert caps.server_side_group_by is False
        assert caps.server_side_order_by is False
        assert caps.native_text_field is False
        assert caps.search_iterator_cursor is False
        assert caps.query_iterator_cursor is False

    def test_the_ceiling_and_page_order_keep_todays_defaults(self, caps):
        """Not "unknown" in these two: the page loop already assumes
        both, and re-checks page order at runtime anyway."""
        assert caps.per_call_row_ceiling == ROW_CEILING_2X
        assert caps.ordered_iterator_pages is True


class TestDefaults:
    def test_client_release_defaults_to_the_installed_pymilvus(self):
        assert (
            capabilities_for("v2.6.4").client_release
            == installed_client_release()
        )

    def test_the_table_is_frozen(self):
        """A capability set is a fact about a connection, not a knob:
        code that wants different behaviour changes the SQL or the
        server, it does not flip a flag on a shared object."""
        caps = capabilities_for("v2.6.4")
        # Through `setattr` so the deliberate error stays a *runtime*
        # assertion: spelled as a plain assignment, the type checker
        # rejects the test file itself.
        with pytest.raises(AttributeError):
            setattr(caps, "server_side_group_by", True)  # noqa: B010


class StubClient:
    """Stands in for ``MilvusClient``/``AsyncMilvusClient`` so the
    probe can be exercised with no server: the real constructors open
    a channel eagerly, which would make this an integration test."""

    def __init__(self, version="v2.6.4", raises=None, **_kwargs):
        self.version = version
        self.raises = raises
        self.probes = 0

    def get_server_version(self, **_kwargs):
        self.probes += 1
        if self.raises is not None:
            raise self.raises
        return self.version

    def close(self):
        pass


class AsyncStubClient(StubClient):
    async def get_server_version(self, **_kwargs):
        return super().get_server_version()

    async def close(self):
        pass


@pytest.fixture
def stub_client(monkeypatch):
    """Swaps the client class a ``connect()`` builds and hands the
    stub back, so a test can count the probes that actually went out."""

    def install(module, attribute, cls, **kwargs):
        client = cls(**kwargs)
        monkeypatch.setattr(module, attribute, lambda **_: client)
        return client

    return install


class TestConnectionProbe:
    def test_the_version_rpc_happens_once_per_connection(self, stub_client):
        client = stub_client(
            connection_module, "MilvusClient", StubClient, version="v3.0.1"
        )
        conn = milvusql.connect(uri="http://nowhere:19530")
        first = conn.capabilities()

        assert first.server_release == (3, 0)
        assert conn.capabilities() is first
        assert client.probes == 1

    def test_a_failed_probe_raises_a_dbapi_error(self, stub_client):
        stub_client(
            connection_module,
            "MilvusClient",
            StubClient,
            raises=MilvusException(message="no route to server"),
        )
        conn = milvusql.connect(uri="http://nowhere:19530")

        # A raw MilvusException crossing a PEP 249 surface is the thing
        # `dbapi.errors` exists to prevent -- the probe is no exception
        # to that just because it is not a statement.
        with pytest.raises(milvusql.Error):
            conn.capabilities()

    async def test_the_async_connection_mirrors_it(self, stub_client):
        client = stub_client(
            aio,
            "AsyncMilvusClient",
            AsyncStubClient,
            version="milvus_lite-3.2.0",
        )
        conn = aio.connect(uri="http://nowhere:19530")
        first = await conn.capabilities()

        assert first.embedded is True
        assert await conn.capabilities() is first
        assert client.probes == 1
