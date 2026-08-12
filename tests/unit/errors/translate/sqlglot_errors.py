"""D8 error-mapping rows for sqlglot's own parse-time exceptions."""

from __future__ import annotations

import pytest
from sqlglot.errors import ParseError, TokenError, UnsupportedError

import milvusql
from milvusql.dbapi import errors

pytestmark = [pytest.mark.unit, pytest.mark.errors]


def test_parse_error_maps_to_programming_error():
    assert isinstance(
        errors.translate(ParseError("bad sql")), milvusql.ProgrammingError
    )


def test_token_error_maps_to_programming_error():
    assert isinstance(
        errors.translate(TokenError("bad token")), milvusql.ProgrammingError
    )


def test_unsupported_error_maps_to_not_supported_error():
    assert isinstance(
        errors.translate(UnsupportedError("no can do")),
        milvusql.NotSupportedError,
    )
