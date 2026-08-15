"""The SQL this dialect emits for joins, grouped aggregates and
subqueries has to be SQL the DBAPI's relational planner accepts.

Nothing here is dialect code -- the planner lives in `milvusql` and the
dialect inherits the capability for free. What these tests pin down is
the *handshake*: SQLAlchemy's own spelling (labelled projections,
qualified columns, `anon_1` subquery aliases) planning into the reads it
should, and no more of them than it should. Entirely offline: compiling
a statement and planning a `Call` both do zero network I/O."""

from __future__ import annotations

import typing as t

import pytest
import sqlglot
from milvusql_sqlalchemy.dialect import MilvusDialect
from pymilvus import MilvusClient
from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    func,
    select,
)

import milvusql
from milvusql.translate.ast_to_pymilvus import build_call

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]

#: Same deliberate cast the core suite uses: every DQL builder reads
#: nothing off the client.
_client = t.cast("MilvusClient", MilvusClient)

metadata = MetaData()
categories = Table(
    "categories",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("title", String(64)),
)
items = Table(
    "items",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("cat_id", BigInteger, ForeignKey("categories.id")),
    Column("price", Integer),
)


def _plan(statement):
    """Compile through the real dialect, then plan the emitted text."""
    sql = str(
        statement.compile(
            dialect=MilvusDialect(), compile_kwargs={"literal_binds": True}
        )
    )
    return sql, build_call(_client, sqlglot.parse_one(sql, read="milvus"), {})


def _chain(call, raws):
    calls = [call]
    for index in range(1, len(raws)):
        call = call.then(raws[index - 1])
        calls.append(call)
    return calls, call.postprocess(raws[-1])


class TestJoin:
    def test_a_join_plans_one_read_per_table(self):
        _sql, call = _plan(
            select(items.c.id, categories.c.title).join(
                categories, items.c.cat_id == categories.c.id
            )
        )
        calls, (rows, _d, _rc, _l) = _chain(
            call,
            [
                [{"id": 1, "cat_id": 10}],
                [{"id": 10, "title": "book"}],
            ],
        )
        assert [c.kwargs["collection_name"] for c in calls] == [
            "items",
            "categories",
        ]
        assert rows == [(1, "book")]

    def test_an_outer_join_keeps_unmatched_rows(self):
        _sql, call = _plan(
            select(items.c.id, categories.c.title).outerjoin(
                categories, items.c.cat_id == categories.c.id
            )
        )
        _calls, (rows, _d, _rc, _l) = _chain(
            call, [[{"id": 1, "cat_id": 99}], []]
        )
        assert rows == [(1, None)]

    def test_a_filter_on_one_table_reaches_milvus(self):
        _sql, call = _plan(
            select(items.c.id)
            .join(categories, items.c.cat_id == categories.c.id)
            .where(categories.c.title == "book")
        )
        assert call.kwargs["collection_name"] == "categories"
        assert call.kwargs["filter"] == 'title == "book"'


class TestGroupBy:
    def test_group_by_with_a_labelled_aggregate(self):
        _sql, call = _plan(
            select(items.c.cat_id, func.count().label("n")).group_by(
                items.c.cat_id
            )
        )
        rows, desc, _rc, _l = call.postprocess(
            [{"cat_id": 1}, {"cat_id": 1}, {"cat_id": 2}]
        )
        assert rows == [(1, 2), (2, 1)]
        assert [column[0] for column in desc] == ["cat_id", "n"]

    def test_having(self):
        _sql, call = _plan(
            select(items.c.cat_id, func.count().label("n"))
            .group_by(items.c.cat_id)
            .having(func.count() > 1)
        )
        rows, _desc, _rc, _l = call.postprocess(
            [{"cat_id": 1}, {"cat_id": 1}, {"cat_id": 2}]
        )
        assert rows == [(1, 2)]


class TestSubquery:
    def test_an_in_subquery_plans_both_reads(self):
        _sql, call = _plan(
            select(items.c.id).where(
                items.c.cat_id.in_(
                    select(categories.c.id).where(categories.c.title == "book")
                )
            )
        )
        calls, (rows, _d, _rc, _l) = _chain(
            call,
            [
                [{"id": 10}],
                [{"id": 1, "cat_id": 10}, {"id": 2, "cat_id": 99}],
            ],
        )
        assert [c.kwargs["collection_name"] for c in calls] == [
            "categories",
            "items",
        ]
        assert rows == [(1,)]

    def test_the_legacy_count_idiom_stays_a_single_server_side_count(self):
        """``select(count()).select_from(<subquery>)`` -- what
        ``Query.count()`` wraps everything in -- must not start using
        the relational engine now that subqueries are supported: Milvus
        computes this one itself, with no row fetch at all."""
        _sql, call = _plan(
            select(func.count()).select_from(
                select(items.c.id).where(items.c.price > 10).subquery()
            )
        )
        assert call.method == "query"
        assert call.kwargs["output_fields"] == ["count(*)"]
        assert call.then is None


class TestUnsupported:
    def test_a_correlated_exists_is_rejected_by_name(self):
        """What ``.any()``/``.has()`` on a relationship compiles to."""
        with pytest.raises(milvusql.NotSupportedError, match="correlated"):
            _plan(
                select(items.c.id).where(
                    select(categories.c.id)
                    .where(categories.c.id == items.c.cat_id)
                    .exists()
                )
            )
