"""Error-path coverage for statement kinds ``build_call`` has no
builder for at all -- no natural "happy path" for this action, so
there is no ``act.py`` alongside it."""

from __future__ import annotations

import typing as t

import pytest
from pymilvus import MilvusClient
from sqlglot import exp

import milvusql
from milvusql.translate.ast_to_pymilvus import build_call

pytestmark = [pytest.mark.unit, pytest.mark.translate]

#: Same deliberate, understood cast as ``tests.fixtures.translate``'s
#: ``_client`` -- ``build_call``'s signature wants an instance, but
#: nothing here ever reaches a method that reads ``self``.
_client = t.cast("MilvusClient", MilvusClient)


class TestUnsupportedStatements:
    def test_statement_type_outside_the_dispatch_table_raises(self):
        """A synthetic AST node, not something the milvus grammar
        could ever parse to -- exercises ``build_call``'s final
        fallback for a statement kind it has no builder for at all."""

        class Unhandled(exp.Expression):
            pass

        with pytest.raises(milvusql.NotSupportedError, match="Unhandled"):
            build_call(_client, Unhandled(), {})
