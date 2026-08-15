"""Shared helper for the ``translate`` entity's unit tests:
``translate.ast_to_pymilvus.build_call`` wants a live-ish
``MilvusClient`` instance, but ``create_schema``/``prepare_index_params``
are ``@classmethod`` on ``MilvusClient`` (confirmed directly -- neither
reads ``self``), so the class itself stands in for a client without
opening a connection; every DML/DQL builder never touches ``client``
at all. Parsing goes through the real ``sqlglot.parse_one(...,
read="milvus")`` path, not hand-built AST nodes, so these tests
exercise the exact same shapes production code sees."""

from __future__ import annotations

import typing as t

import pytest
import sqlglot
from pymilvus import MilvusClient

from milvusql.translate.ast_to_pymilvus import build_call

#: Same kind of deliberate, understood cast as
#: ``milvusql_sqlalchemy.dialect``'s ``import_dbapi``.
_client = t.cast("MilvusClient", MilvusClient)


def _build(sql: str, parameters: dict | None = None):
    ast = sqlglot.parse_one(sql, read="milvus")
    return build_call(_client, ast, parameters or {})


@pytest.fixture
def build_call_helper():
    """Parse ``sql`` with the real MilvusQL grammar and dispatch it
    through ``build_call``, returning the resulting ``Call``."""
    return _build


def _drive(call, raws: list):
    """Run a ``Call`` chain exactly the way ``Cursor._run`` does --
    ``then`` builds the next call from the previous raw result, and the
    last call's ``postprocess`` turns its own raw result into rows.

    Returns ``(calls, postprocess_result)`` so a test can assert both
    what was sent (one entry per RPC, including any filter a later scan
    had join keys pushed into) and what came back."""
    calls = [call]
    for index in range(1, len(raws)):
        call = call.then(raws[index - 1])
        calls.append(call)
    return calls, call.postprocess(raws[-1])


@pytest.fixture
def drive_chain():
    """Drive a multi-RPC ``Call`` chain over canned pymilvus results."""
    return _drive
