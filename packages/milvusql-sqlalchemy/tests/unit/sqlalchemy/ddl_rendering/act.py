"""Unit coverage for ``MilvusDDLCompiler``: ``CREATE TABLE``/
``CREATE INDEX`` rendering (the ``milvusql_*`` ``Table``/``Index``
dialect options). Entirely offline --
``some_stmt.compile(dialect=MilvusDialect())`` does zero network I/O."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.dialect import MilvusDialect
from milvusql_sqlalchemy.types import VECTOR
from sqlalchemy import BigInteger, Column, MetaData, String, Table
from sqlalchemy.schema import CreateIndex, CreateTable
from sqlalchemy.sql.schema import Index

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]


def _normalize(sql: str) -> str:
    """Collapses the base ``DDLCompiler``'s multi-line/tab formatting
    to single spaces so the assertion checks content, not incidental
    whitespace."""
    return " ".join(sql.split())


@pytest.fixture
def dialect() -> MilvusDialect:
    return MilvusDialect()


class TestCreateTableDDL:
    def test_inline_primary_key_autoincrement_and_table_options(self, dialect):
        metadata = MetaData()
        items = Table(
            "items",
            metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True),
            Column("category", String(64)),
            Column("embedding", VECTOR(8)),
            milvusql_shards=2,
            milvusql_consistency_level="Bounded",
            milvusql_partition_key="category",
        )
        sql = _normalize(str(CreateTable(items).compile(dialect=dialect)))
        assert sql == (
            "CREATE TABLE items ( "
            "id BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT, "
            "category VARCHAR(64), "
            "embedding VECTOR(8), "
            "PRIMARY KEY (id) ) "
            "WITH (shards=2, consistency_level='Bounded', "
            "partition_key='category')"
        )

    def test_varchar_primary_key_without_autoincrement_keyword(self, dialect):
        """Only the ``Table``'s designated autoincrement column gets
        ``AUTO_INCREMENT``; a ``VARCHAR`` primary key never does."""
        metadata = MetaData()
        items = Table(
            "items",
            metadata,
            Column("id", String(36), primary_key=True),
            Column("val", String(8)),
        )
        sql = _normalize(str(CreateTable(items).compile(dialect=dialect)))
        assert sql == (
            "CREATE TABLE items ( "
            "id VARCHAR(36) NOT NULL PRIMARY KEY, "
            "val VARCHAR(8), "
            "PRIMARY KEY (id) )"
        )

    def test_no_dialect_options_omits_the_with_clause(self, dialect):
        metadata = MetaData()
        items = Table(
            "items", metadata, Column("id", BigInteger, primary_key=True)
        )
        sql = _normalize(str(CreateTable(items).compile(dialect=dialect)))
        assert sql == (
            "CREATE TABLE items ( "
            "id BIGINT NOT NULL PRIMARY KEY AUTO_INCREMENT, "
            "PRIMARY KEY (id) )"
        )


class TestCreateIndexDDL:
    def test_using_and_with_options_render_as_suffixes(self, dialect):
        metadata = MetaData()
        items = Table(
            "items",
            metadata,
            Column("id", BigInteger, primary_key=True),
            Column("embedding", VECTOR(8)),
        )
        index = Index(
            "idx_emb",
            items.c.embedding,
            milvusql_using="HNSW",
            milvusql_with={
                "metric_type": "COSINE",
                "M": 16,
                "ef_construction": 200,
            },
        )
        sql = _normalize(str(CreateIndex(index).compile(dialect=dialect)))
        assert sql == (
            "CREATE INDEX idx_emb ON items (embedding) USING HNSW "
            "WITH (metric_type='COSINE', M=16, ef_construction=200)"
        )

    def test_no_options_omits_using_and_with(self, dialect):
        metadata = MetaData()
        items = Table(
            "items",
            metadata,
            Column("id", BigInteger, primary_key=True),
            Column("embedding", VECTOR(8)),
        )
        index = Index("idx_plain", items.c.embedding)
        sql = _normalize(str(CreateIndex(index).compile(dialect=dialect)))
        assert sql == "CREATE INDEX idx_plain ON items (embedding)"
