"""D8 error-mapping rows for transport-level gRPC errors -- some of
these (a bare transport-level ``UNAVAILABLE``) aren't reachable
through a real ``execute()`` call in this suite (Milvus Lite doesn't
fail that way), so they're constructed directly here rather than
skipped."""

from __future__ import annotations

import grpc
import pytest

import milvusql
from milvusql.dbapi import errors

pytestmark = [pytest.mark.unit, pytest.mark.errors]


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


def test_grpc_deadline_exceeded_maps_to_operational_error():
    exc = _fake_rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED)
    assert isinstance(errors.translate(exc), milvusql.OperationalError)


def test_grpc_error_without_call_code_falls_back_to_database_error():
    """The base ``grpc.RpcError`` declares no ``.code()`` at all --
    ``_translate_grpc_error`` only calls it when the exception is also
    a ``grpc.Call`` (real gRPC errors always are); a bare instance of
    the base class itself must fall back rather than raise."""
    assert isinstance(
        errors.translate(grpc.RpcError()), milvusql.DatabaseError
    )
