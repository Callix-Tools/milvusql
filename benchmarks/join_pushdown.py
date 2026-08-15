"""Benchmark: what the relational planner's pushdown actually buys.

Three questions, each answered by timing the same logical query with
and without the optimization that makes it cheap:

1. **Key pushdown.** An ANN search returning ``TOP_K`` hits joined
   against a large collection should read ``TOP_K`` rows from it, not
   all of them -- the join keys learned from the search are pushed into
   the second read as ``key in [...]``.
2. **Predicate pushdown.** A selective ``WHERE`` conjunct naming one
   collection travels to Milvus as that collection's own ``filter``;
   the join then runs over the few surviving rows.
3. **Baseline.** The same vector search with no join at all, so the
   join overhead is visible next to what Milvus itself costs.

Usage::

    uv run python benchmarks/join_pushdown.py                    # Milvus Lite
    MILVUS_URI=http://localhost:19530 \\
        uv run python benchmarks/join_pushdown.py                # real server

Row counts are configurable; defaults stay under Milvus's per-call
ceiling so the run also works against Milvus Lite (which cannot serve
the primary-key-ordered pages the unbounded reads use past it)::

    ITEMS_ROWS=100000 CATS_ROWS=5000 uv run python benchmarks/...
"""

from __future__ import annotations

import contextlib
import os
import random
import statistics
import time

import milvusql

ITEMS_ROWS = int(os.getenv("ITEMS_ROWS", "10000"))
CATS_ROWS = int(os.getenv("CATS_ROWS", "1000"))
DIM = int(os.getenv("DIM", "64"))
TOP_K = int(os.getenv("TOP_K", "50"))
REPEAT = int(os.getenv("REPEAT", "7"))


def _connect():
    uri = os.getenv("MILVUS_URI", "./benchmark.db")
    token = os.getenv("MILVUS_TOKEN", "")
    return milvusql.connect(uri=uri, token=token)


def _seed(cur) -> None:
    for table in ("bench_items", "bench_cats"):
        with contextlib.suppress(milvusql.Error):
            cur.execute(f"DROP TABLE {table}")
    cur.execute(
        "CREATE TABLE bench_cats (id BIGINT PRIMARY KEY, title VARCHAR(64), "
        "budget DOUBLE, v VECTOR(2))"
    )
    cur.execute(
        "CREATE INDEX idx_bc ON bench_cats (v) USING FLAT "
        "WITH (metric_type='L2')"
    )
    cur.execute(
        "CREATE TABLE bench_items (id BIGINT PRIMARY KEY, cat_id BIGINT, "
        f"price DOUBLE, embedding VECTOR({DIM}))"
    )
    cur.execute(
        "CREATE INDEX idx_bi ON bench_items (embedding) USING HNSW "
        "WITH (metric_type='L2', M=16, ef_construction=200)"
    )
    rng = random.Random(1313)
    cur.executemany(
        "INSERT INTO bench_cats (id, title, budget, v) "
        "VALUES (:id, :title, :budget, :v)",
        [
            {
                "id": index,
                "title": f"cat-{index}",
                "budget": rng.uniform(1, 100),
                "v": [0.0, 0.0],
            }
            for index in range(CATS_ROWS)
        ],
    )
    batch = 5000
    for start in range(0, ITEMS_ROWS, batch):
        cur.executemany(
            "INSERT INTO bench_items (id, cat_id, price, embedding) "
            "VALUES (:id, :cat_id, :price, :embedding)",
            [
                {
                    "id": index,
                    "cat_id": index % CATS_ROWS,
                    "price": rng.uniform(1, 1000),
                    "embedding": [rng.random() for _ in range(DIM)],
                }
                for index in range(start, min(start + batch, ITEMS_ROWS))
            ],
        )
    cur.execute("LOAD TABLE bench_items")
    cur.execute("LOAD TABLE bench_cats")


def _timed(cur, label: str, sql: str, params: dict) -> None:
    samples = []
    rows = 0
    for _ in range(REPEAT):
        started = time.perf_counter()
        cur.execute(sql, params)
        rows = len(cur.fetchall())
        samples.append((time.perf_counter() - started) * 1000)
    median = statistics.median(samples)
    spread = (
        f"p50={median:8.2f}ms  min={min(samples):8.2f}ms  "
        f"max={max(samples):8.2f}ms"
    )
    print(f"{label:58} {spread}  rows={rows}")


def main() -> None:
    conn = _connect()
    cur = conn.cursor()
    print(
        f"seeding: bench_items={ITEMS_ROWS} rows (dim={DIM}), "
        f"bench_cats={CATS_ROWS} rows ..."
    )
    _seed(cur)
    query_vector = [0.5] * DIM

    _timed(
        cur,
        f"1. ANN top-{TOP_K} (no join, baseline)",
        f"SELECT id FROM bench_items ORDER BY embedding <-> :q LIMIT {TOP_K}",
        {"q": query_vector},
    )
    _timed(
        cur,
        f"2. ANN top-{TOP_K} JOIN cats (key pushdown: reads top-k + <=k)",
        "SELECT i.id, c.title FROM bench_items AS i "
        "JOIN bench_cats AS c ON i.cat_id = c.id "
        f"ORDER BY i.embedding <-> :q LIMIT {TOP_K}",
        {"q": query_vector},
    )
    _timed(
        cur,
        "3. selective WHERE JOIN (predicate pushdown)",
        "SELECT i.id, c.title FROM bench_items AS i "
        "JOIN bench_cats AS c ON i.cat_id = c.id WHERE i.price > :floor",
        {"floor": 995.0},
    )
    _timed(
        cur,
        "4. GROUP BY over the join (full read of both sides)",
        "SELECT c.title, COUNT(*) AS n FROM bench_items AS i "
        "JOIN bench_cats AS c ON i.cat_id = c.id "
        "WHERE i.price > :floor GROUP BY c.title ORDER BY n DESC LIMIT 10",
        {"floor": 500.0},
    )
    conn.close()


if __name__ == "__main__":
    main()
