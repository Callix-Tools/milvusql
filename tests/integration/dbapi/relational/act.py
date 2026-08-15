"""Integration coverage for ``JOIN``/``GROUP BY``/subqueries against a
real Milvus server.

The unit tests drive the ``Call`` chain over canned results, which
proves the planning and the combination. What only a real server can
show is the part in between: that the pushed-down filters and the
``key in [...]`` restriction are expressions Milvus actually accepts,
that two collections really are read in one ``execute()``, and that the
async cursor answers identically -- it shares this module verbatim, and
that is worth verifying rather than asserting."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.dbapi]

CREATE_CATEGORIES = (
    "CREATE TABLE categories ("
    "id BIGINT PRIMARY KEY, "
    "v VECTOR(2), "
    "title VARCHAR(64)"
    ") WITH (shards=1, consistency_level='Strong')"
)
CREATE_CATEGORIES_INDEX = (
    "CREATE INDEX idx_v ON categories (v) USING HNSW "
    "WITH (metric_type='COSINE')"
)
CREATE_ITEMS_WITH_FK = (
    "CREATE TABLE joined_items ("
    "id BIGINT PRIMARY KEY AUTO_INCREMENT, "
    "embedding VECTOR(8), "
    "cat_id BIGINT, "
    "price DOUBLE"
    ") WITH (shards=1, consistency_level='Strong')"
)
CREATE_ITEMS_WITH_FK_INDEX = (
    "CREATE INDEX idx_emb2 ON joined_items (embedding) USING HNSW "
    "WITH (metric_type='COSINE')"
)

ITEMS = [
    {"emb": [0.1] * 8, "cat_id": 1, "price": 10.0},
    {"emb": [0.2] * 8, "cat_id": 1, "price": 30.0},
    {"emb": [0.9] * 8, "cat_id": 2, "price": 50.0},
    # points at a category that does not exist -- the LEFT JOIN orphan
    {"emb": [0.5] * 8, "cat_id": 99, "price": 70.0},
]


@pytest.fixture
def two_collections(conn, cur):
    cur.execute(CREATE_CATEGORIES)
    cur.execute(CREATE_CATEGORIES_INDEX)
    cur.execute(CREATE_ITEMS_WITH_FK)
    cur.execute(CREATE_ITEMS_WITH_FK_INDEX)
    cur.executemany(
        "INSERT INTO categories (id, v, title) VALUES (:id, :v, :title)",
        [
            {"id": 1, "v": [0.1, 0.2], "title": "book"},
            {"id": 2, "v": [0.3, 0.4], "title": "movie"},
            {"id": 3, "v": [0.5, 0.6], "title": "unused"},
        ],
    )
    cur.executemany(
        "INSERT INTO joined_items (embedding, cat_id, price) "
        "VALUES (:emb, :cat_id, :price)",
        ITEMS,
    )
    return conn


class TestJoin:
    def test_inner_join_reads_both_collections(self, two_collections, cur):
        cur.execute(
            "SELECT i.price, c.title FROM joined_items AS i "
            "JOIN categories AS c ON i.cat_id = c.id ORDER BY i.price"
        )
        assert cur.fetchall() == [
            (10.0, "book"),
            (30.0, "book"),
            (50.0, "movie"),
        ]
        assert [column[0] for column in cur.description] == ["price", "title"]

    def test_left_join_keeps_the_unmatched_row(self, two_collections, cur):
        cur.execute(
            "SELECT i.price, c.title FROM joined_items AS i "
            "LEFT JOIN categories AS c ON i.cat_id = c.id ORDER BY i.price"
        )
        assert cur.fetchall() == [
            (10.0, "book"),
            (30.0, "book"),
            (50.0, "movie"),
            (70.0, None),
        ]

    def test_a_pushed_down_filter_is_accepted_by_milvus(
        self, two_collections, cur
    ):
        """The per-collection conjunct becomes a real Milvus filter
        expression -- this is the test that it parses server-side."""
        cur.execute(
            "SELECT i.price, c.title FROM joined_items AS i "
            "JOIN categories AS c ON i.cat_id = c.id "
            "WHERE i.price > :floor AND c.title = :title",
            {"floor": 20.0, "title": "book"},
        )
        assert cur.fetchall() == [(30.0, "book")]

    def test_a_vector_search_joined_to_metadata(self, two_collections, cur):
        cur.execute(
            "SELECT i.price, c.title FROM joined_items AS i "
            "JOIN categories AS c ON i.cat_id = c.id "
            "ORDER BY i.embedding <=> :q LIMIT 2",
            {"q": [0.1] * 8},
        )
        rows = cur.fetchall()
        assert len(rows) <= 2
        assert all(row[1] in {"book", "movie"} for row in rows)


class TestGroupBy:
    def test_grouped_aggregate(self, two_collections, cur):
        cur.execute(
            "SELECT cat_id, COUNT(*) AS n, SUM(price) AS total "
            "FROM joined_items GROUP BY cat_id ORDER BY cat_id"
        )
        assert cur.fetchall() == [
            (1, 2, 40.0),
            (2, 1, 50.0),
            (99, 1, 70.0),
        ]

    def test_having_and_limit(self, two_collections, cur):
        cur.execute(
            "SELECT cat_id, COUNT(*) AS n FROM joined_items "
            "GROUP BY cat_id HAVING COUNT(*) > 1"
        )
        assert cur.fetchall() == [(1, 2)]


class TestSubquery:
    def test_in_subquery_over_a_second_collection(self, two_collections, cur):
        cur.execute(
            "SELECT price FROM joined_items WHERE cat_id IN "
            "(SELECT id FROM categories WHERE title = :t) ORDER BY price",
            {"t": "book"},
        )
        assert cur.fetchall() == [(10.0,), (30.0,)]

    def test_scalar_subquery(self, two_collections, cur):
        cur.execute(
            "SELECT price FROM joined_items WHERE price > "
            "(SELECT AVG(price) FROM joined_items) ORDER BY price"
        )
        assert cur.fetchall() == [(50.0,), (70.0,)]


class TestAsyncMatchesSync:
    async def test_the_async_cursor_answers_identically(
        self, two_collections, cur, aconn
    ):
        """``relational`` is shared verbatim by both cursors: the chain
        it builds is awaited by one and called directly by the other,
        so the same MilvusQL text must produce the same rows."""
        sql = (
            "SELECT i.price, c.title FROM joined_items AS i "
            "JOIN categories AS c ON i.cat_id = c.id ORDER BY i.price"
        )
        cur.execute(sql)
        expected = cur.fetchall()

        acur = aconn.cursor()
        await acur.execute(sql)
        assert await acur.fetchall() == expected

    async def test_the_async_cursor_groups_identically(
        self, two_collections, cur, aconn
    ):
        sql = (
            "SELECT cat_id, COUNT(*) AS n FROM joined_items "
            "GROUP BY cat_id ORDER BY cat_id"
        )
        cur.execute(sql)
        expected = cur.fetchall()

        acur = aconn.cursor()
        await acur.execute(sql)
        assert await acur.fetchall() == expected
