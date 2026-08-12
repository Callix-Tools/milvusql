"""One test per D8 error-mapping row. Unit-level, against
``errors.translate()`` directly: some upstream exception types (a
generation-time ``UnsupportedError``, a transport-level
``UNAVAILABLE``) aren't reachable through a real ``execute()`` call in
this suite (Milvus Lite doesn't fail that way), so they're constructed
directly here rather than skipped. The two rows that *are* naturally
reachable end-to-end (``ProgrammingError`` via a missing collection,
``NotSupportedError`` via ``rollback()``) are covered live, sync and
async, in ``test_dbapi.py`` -- not repeated here."""

from __future__ import annotations

import grpc
from pymilvus.exceptions import (
    CollectionNotExistException,
    ConnectError,
    ErrorCode,
    IndexNotExistException,
    MilvusException,
    MilvusUnavailableException,
)
from sqlglot.errors import ParseError, TokenError, UnsupportedError

import milvusql
from milvusql.dbapi import errors


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
    """D8's amendment: the server reports "collection doesn't exist" as
    a bare ``MilvusException`` with ``code=100``, not a
    ``CollectionNotExistException`` -- confirmed against a real Milvus
    Lite instance (see ``test_dbapi.py``'s
    ``test_missing_collection_raises_programming_error``)."""
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


def _fake_rpc_error(status: grpc.StatusCode) -> grpc.RpcError:
    """A minimal stand-in for a real gRPC error. Registered as a
    virtual subclass of ``grpc.Call`` (real errors -- ``_InactiveRpcError``,
    ``AioRpcError`` -- multiply-inherit both) rather than implementing
    every one of ``grpc.Call``'s eight abstract methods just to satisfy
    ``isinstance``."""

    class FakeRpcError(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return status

    grpc.Call.register(FakeRpcError)
    return FakeRpcError()


def test_grpc_unimplemented_maps_to_not_supported_error():
    exc = _fake_rpc_error(grpc.StatusCode.UNIMPLEMENTED)
    assert isinstance(errors.translate(exc), milvusql.NotSupportedError)


def test_grpc_unavailable_maps_to_operational_error():
    exc = _fake_rpc_error(grpc.StatusCode.UNAVAILABLE)
    assert isinstance(errors.translate(exc), milvusql.OperationalError)
