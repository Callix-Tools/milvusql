"""A guided, top-to-bottom tour of the ``milvusql`` DBAPI.

Run with:

    python walkthrough.py

Defaults to a local Milvus Lite file (``./walkthrough.db``) so there's
nothing to stand up first. Point it at a real server instead with:

    MILVUS_URI=http://localhost:19530 MILVUS_TOKEN=root:Milvus python walkthrough.py
"""

from __future__ import annotations

import os

import milvusql

MILVUS_URI = os.environ.get("MILVUS_URI", "./walkthrough.db")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")

CATALOG = [
    # (embedding, category, title)
    ([0.10, 0.12, 0.11, 0.09, 0.10, 0.13, 0.11, 0.10], "book", "Dune"),
    ([0.11, 0.10, 0.12, 0.10, 0.09, 0.11, 0.10, 0.12], "book", "Neuromancer"),
    ([0.91, 0.88, 0.90, 0.92, 0.89, 0.91, 0.90, 0.88], "movie", "Blade Runner"),
    ([0.90, 0.92, 0.89, 0.91, 0.90, 0.88, 0.91, 0.90], "movie", "The Matrix"),
]


def step(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    # 1. Connect. `connect()` takes every `MilvusClient` keyword
    # (uri/token/db_name/timeout/...) straight through -- this layer
    # doesn't reinvent Milvus's own connection parameters.
    step("connect")
    conn = milvusql.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    cur = conn.cursor()

    try:
        # 2. Schema. One BIGINT auto-id primary key, one VECTOR column,
        # two scalar columns -- see the root README for the full DDL
        # grammar.
        step("CREATE TABLE")
        try:
            # Best-effort cleanup from a previous crashed run --
            # MilvusQL has no `DROP TABLE IF EXISTS`, so a missing
            # collection is just a `ProgrammingError` to swallow here.
            cur.execute("DROP TABLE walkthrough_items")
        except milvusql.Error:
            pass
        cur.execute(
            "CREATE TABLE walkthrough_items ("
            "id BIGINT PRIMARY KEY AUTO_INCREMENT, "
            "embedding VECTOR(8), "
            "category VARCHAR(64), "
            "title VARCHAR(128)"
            ") WITH (shards=1, consistency_level='Strong')"
        )
        print("collection 'walkthrough_items' created")

        # 3. Index + load. A vector column needs an index before it can
        # be searched or the collection loaded into memory.
        step("CREATE INDEX + LOAD TABLE")
        cur.execute(
            "CREATE INDEX idx_embedding ON walkthrough_items (embedding) "
            "USING HNSW WITH (metric_type='COSINE', M=16, ef_construction=200)"
        )
        cur.execute("LOAD TABLE walkthrough_items")
        print("index built, collection loaded")

        # 4. Insert. `executemany()` batches every row into a single
        # `insert()` RPC instead of one round trip per row.
        step("INSERT (batched)")
        cur.executemany(
            "INSERT INTO walkthrough_items (embedding, category, title) "
            "VALUES (:emb, :cat, :title)",
            [
                {"emb": emb, "cat": cat, "title": title}
                for emb, cat, title in CATALOG
            ],
        )
        print(f"inserted {cur.rowcount} rows")

        # 5. Plain filter SELECT -- no vector column referenced, so
        # this compiles to a `query()`, not a `search()`.
        step("SELECT (plain filter)")
        cur.execute(
            "SELECT id, title FROM walkthrough_items "
            "WHERE category = :cat LIMIT 10",
            {"cat": "book"},
        )
        for row in cur.fetchall():
            print(row)

        # 6. Vector search -- `ORDER BY <vector column> <=> :q` compiles
        # to a `search()` instead. `<=>` is cosine distance (matches the
        # index's own `metric_type='COSINE'` above); `<->`/`<#>`/`<+>`
        # are L2/inner-product/L1.
        step("SELECT (vector search)")
        query_vector = [0.10, 0.11, 0.10, 0.10, 0.11, 0.12, 0.10, 0.11]
        cur.execute(
            "SELECT id, title, category FROM walkthrough_items "
            "ORDER BY embedding <=> :q LIMIT 3",
            {"q": query_vector},
        )
        for row in cur.fetchall():
            print(row)

        # 7. Update.
        step("UPDATE")
        cur.execute(
            "UPDATE walkthrough_items SET category = :new "
            "WHERE title = :title",
            {"new": "sci-fi-book", "title": "Dune"},
        )
        print(f"updated {cur.rowcount} row(s)")

        # 8. Delete.
        step("DELETE")
        cur.execute(
            "DELETE FROM walkthrough_items WHERE category = :cat",
            {"cat": "movie"},
        )
        print(f"deleted {cur.rowcount} row(s)")

        step("final state")
        cur.execute("SELECT id, title, category FROM walkthrough_items")
        for row in cur.fetchall():
            print(row)
    finally:
        # 9. Clean up and close. `commit()` is a no-op (every mutation
        # above already applied the moment it returned) -- there's
        # nothing left to commit.
        step("cleanup")
        try:
            cur.execute("DROP TABLE walkthrough_items")
        except milvusql.Error:
            pass
        conn.close()
        print("connection closed")


if __name__ == "__main__":
    main()
