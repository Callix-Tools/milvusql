"""D8 error-mapping rows for pymilvus's own exception types, including
the "generic ``MilvusException`` by error code" amendment: the server
reports "collection doesn't exist" as a bare ``MilvusException`` with
``code=100``, not a ``CollectionNotExistException`` -- confirmed
against a real Milvus Lite instance (see
``tests.integration.dbapi.execute.error``'s
``test_missing_collection_raises_programming_error``)."""

from __future__ import annotations

import pytest
from pymilvus.exceptions import (
    CollectionNotExistException,
    ConnectError,
    ErrorCode,
    IndexNotExistException,
    MilvusException,
    MilvusUnavailableException,
)

import milvusql
from milvusql.dbapi import errors

pytestmark = [pytest.mark.unit, pytest.mark.errors]


def test_milvus_unavailable_maps_to_operational_error():
    assert isinstance(
        errors.translate(MilvusUnavailableException(message="down")),
        milvusql.OperationalError,
    )


def test_connect_error_maps_to_operational_error():
    assert isinstance(
        errors.translate(ConnectError(message="refused")),
        milvusql.OperationalError,
    )


def test_collection_not_exist_maps_to_programming_error():
    assert isinstance(
        errors.translate(CollectionNotExistException(message="nope")),
        milvusql.ProgrammingError,
    )


def test_index_not_exist_maps_to_programming_error():
    assert isinstance(
        errors.translate(IndexNotExistException(message="nope")),
        milvusql.ProgrammingError,
    )


def test_generic_milvus_exception_by_error_code_maps_to_programming_error():
    exc = MilvusException(code=ErrorCode.COLLECTION_NOT_FOUND, message="nope")
    assert isinstance(errors.translate(exc), milvusql.ProgrammingError)


def test_not_loaded_message_maps_to_programming_error():
    exc = MilvusException(code=1, message="collection not loaded")
    assert isinstance(errors.translate(exc), milvusql.ProgrammingError)


def test_unrecognized_milvus_exception_falls_back_to_database_error():
    exc = MilvusException(code=1, message="something unexpected")
    result = errors.translate(exc)
    assert isinstance(result, milvusql.DatabaseError)
    assert not isinstance(result, milvusql.ProgrammingError)
