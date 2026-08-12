"""Unit coverage for ``DatabaseWrapper.get_connection_params``: building
``MilvusClient`` connect kwargs out of one ``DATABASES`` entry. Doesn't
need ``settings.configure()``/``django.setup()`` -- ``DatabaseWrapper``
only reads its own ``settings_dict``."""

from __future__ import annotations

import pytest
from milvusql_django.base import DatabaseWrapper

pytestmark = [pytest.mark.unit, pytest.mark.django]


def _wrapper(**overrides):
    settings_dict = {
        "NAME": "",
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
    settings_dict.update(overrides)
    return DatabaseWrapper(settings_dict, alias="default")


class TestGetConnectionParams:
    def test_bare_name_yields_a_lite_uri_with_no_db_name_or_token(self):
        wrapper = _wrapper(NAME="/var/lib/milvus/milvus.db")
        assert wrapper.get_connection_params() == {
            "uri": "/var/lib/milvus/milvus.db"
        }

    def test_host_set_builds_an_http_uri_and_carries_name_as_db_name(self):
        wrapper = _wrapper(NAME="mydb", HOST="myhost", PORT=19531)
        assert wrapper.get_connection_params() == {
            "uri": "http://myhost:19531",
            "db_name": "mydb",
        }

    def test_options_secure_true_switches_the_uri_scheme_to_https(self):
        wrapper = _wrapper(
            NAME="mydb", HOST="myhost", PORT=19531, OPTIONS={"secure": True}
        )
        assert wrapper.get_connection_params() == {
            "uri": "https://myhost:19531",
            "db_name": "mydb",
            "secure": True,
        }

    def test_user_and_password_join_into_one_token(self):
        # Milvus's own documented default credential (see
        # `token_from_credentials`'s docstring), not a real secret.
        wrapper = _wrapper(
            NAME="mydb",
            USER="root",
            PASSWORD="Milvus",  # noqa: S106
        )
        assert wrapper.get_connection_params() == {
            "uri": "mydb",
            "token": "root:Milvus",
        }

    def test_password_only_passes_through_as_the_token_unchanged(self):
        wrapper = _wrapper(NAME="mydb", PASSWORD="root:Milvus")  # noqa: S106
        assert wrapper.get_connection_params() == {
            "uri": "mydb",
            "token": "root:Milvus",
        }

    def test_options_dict_passes_through_into_the_returned_params(self):
        wrapper = _wrapper(NAME="mydb", OPTIONS={"timeout": 5})
        assert wrapper.get_connection_params() == {"uri": "mydb", "timeout": 5}
