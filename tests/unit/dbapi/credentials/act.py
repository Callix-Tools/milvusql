"""Unit coverage for the pure helpers in ``dbapi.connection`` that
don't need a live client -- currently just ``token_from_credentials``,
the one both ``milvusql-sqlalchemy``'s and ``milvusql-django``'s
connect-arg builders call to turn a split user/password into the
single ``"user:password"`` token Milvus actually wants. No natural
error branch: every case here is a valid input/output pair."""

from __future__ import annotations

import pytest

from milvusql.dbapi.connection import token_from_credentials

pytestmark = [pytest.mark.unit, pytest.mark.dbapi]


def test_user_and_password_join_into_one_token():
    assert token_from_credentials("user", "pass") == "user:pass"


def test_password_only_passes_through_unchanged():
    """Covers a full ``"user:token"`` pasted directly into a URL's
    password slot, with no separate username given."""
    assert token_from_credentials(None, "root:Milvus") == "root:Milvus"


def test_user_only_without_a_password_yields_no_token():
    assert token_from_credentials("user", None) is None


def test_neither_user_nor_password_yields_no_token():
    assert token_from_credentials(None, None) is None


def test_empty_string_user_is_treated_as_absent():
    assert token_from_credentials("", "pass") == "pass"
