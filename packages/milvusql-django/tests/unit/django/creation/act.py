"""``DatabaseCreation`` is deliberately thin (D11 spike scope) --
unit coverage for its explicit no-ops, which exist only so the test
runner doesn't crash with a confusing SQL error rather than to
actually manage a test database."""

from __future__ import annotations

import pytest
from milvusql_django.base import DatabaseWrapper
from milvusql_django.creation import DatabaseCreation

pytestmark = [pytest.mark.unit, pytest.mark.django]


@pytest.fixture
def creation():
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
        # `_get_test_db_name` (Django's base implementation, unmodified
        # here) reads this directly -- normally filled in by
        # `ConnectionHandler`'s settings normalization, which a
        # hand-built settings_dict bypasses.
        "TEST": {"NAME": None},
    }
    wrapper = DatabaseWrapper(settings_dict, alias="default")
    return DatabaseCreation(wrapper)


def test_create_test_db_returns_the_computed_test_db_name(creation):
    assert (
        creation._create_test_db(verbosity=0, autoclobber=True) == "test_mydb"
    )


def test_destroy_test_db_is_a_noop(creation):
    assert creation._destroy_test_db("test_mydb", verbosity=0) is None


def test_sql_table_creation_suffix_is_empty(creation):
    assert creation.sql_table_creation_suffix() == ""
