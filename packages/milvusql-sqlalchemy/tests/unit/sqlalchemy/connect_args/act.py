"""Unit coverage for ``MilvusDialect.create_connect_args``: connect-arg
assembly from a parsed ``URL`` -- every case here is a valid URL shape,
so there's no natural error case for this action."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.dialect import MilvusDialect
from sqlalchemy.engine.url import make_url

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]


@pytest.fixture
def dialect() -> MilvusDialect:
    return MilvusDialect()


class TestCreateConnectArgs:
    def test_no_host_uses_the_database_path_as_uri(self, dialect):
        url = make_url("milvusql:////var/lib/foo/milvus.db")
        assert dialect.create_connect_args(url) == (
            [],
            {"uri": "/var/lib/foo/milvus.db"},
        )

    def test_no_host_and_no_database_yields_an_empty_uri(self, dialect):
        url = make_url("milvusql://")
        assert dialect.create_connect_args(url) == ([], {"uri": ""})

    def test_host_builds_an_http_uri_with_default_port_and_db_name(
        self, dialect
    ):
        url = make_url("milvusql://host/mydb")
        assert dialect.create_connect_args(url) == (
            [],
            {"uri": "http://host:19530", "db_name": "mydb"},
        )

    def test_explicit_port_overrides_the_default(self, dialect):
        url = make_url("milvusql://host:19531/mydb")
        assert dialect.create_connect_args(url) == (
            [],
            {"uri": "http://host:19531", "db_name": "mydb"},
        )

    def test_secure_query_param_switches_the_scheme_to_https(self, dialect):
        url = make_url("milvusql://host:19530/mydb?secure=true")
        assert dialect.create_connect_args(url) == (
            [],
            {"uri": "https://host:19530", "db_name": "mydb"},
        )

    def test_username_and_password_combine_into_one_token(self, dialect):
        url = make_url("milvusql://user:pass@host:19530/mydb")
        assert dialect.create_connect_args(url) == (
            [],
            {
                "uri": "http://host:19530",
                "db_name": "mydb",
                "token": "user:pass",
            },
        )

    def test_password_only_token_is_forwarded_verbatim(self, dialect):
        """A full ``"user:token"`` pasted directly into the password
        slot, with no separate username in the URL."""
        url = make_url("milvusql://:root:Milvus@host:19530/mydb")
        assert dialect.create_connect_args(url) == (
            [],
            {
                "uri": "http://host:19530",
                "db_name": "mydb",
                "token": "root:Milvus",
            },
        )

    def test_username_without_password_yields_no_token_key(self, dialect):
        url = make_url("milvusql://user@host:19530/mydb")
        kwargs = dialect.create_connect_args(url)[1]
        assert "token" not in kwargs

    def test_neither_username_nor_password_yields_no_token_key(self, dialect):
        url = make_url("milvusql://host:19530/mydb")
        kwargs = dialect.create_connect_args(url)[1]
        assert "token" not in kwargs
