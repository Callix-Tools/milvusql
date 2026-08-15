# Benchmarks

What the relational planner's pushdown actually buys, measured through
the public DBAPI (`milvusql.connect()` → `cursor.execute()`), not
through internals.

## Why these four queries

Milvus reads one collection per RPC and joins nothing — every JOIN /
GROUP BY is planned into one read per collection plus a client-side
Polars evaluation. The whole design stands or falls on two pushdowns:

| # | Query | What it proves |
|---|---|---|
| 1 | ANN top-k, no join | Baseline: what Milvus itself costs |
| 2 | ANN top-k `JOIN` cats | **Key pushdown**: the second collection is read with `id in [...]` learned from the search — `top_k` rows, never the whole collection. The join costs milliseconds over the baseline, independent of the joined collection's size |
| 3 | selective `WHERE` + `JOIN` | **Predicate pushdown**: the filter travels to Milvus, the join runs over the survivors only |
| 4 | `GROUP BY` over a broad join | The honest worst case: a wide filter means a wide read; the grouping itself is Polars and cheap, the read dominates |

## Running

```bash
# Milvus Lite (zero setup, illustrative numbers only)
uv run python benchmarks/join_pushdown.py

# Real Milvus server (the numbers that matter)
MILVUS_URI=http://localhost:19530 ITEMS_ROWS=100000 CATS_ROWS=5000 \
    uv run python benchmarks/join_pushdown.py
```

Knobs: `ITEMS_ROWS`, `CATS_ROWS`, `DIM`, `TOP_K`, `REPEAT`. Each query
runs `REPEAT` times; p50/min/max wall-clock are reported. Defaults stay
under Milvus's 16384-row per-call ceiling so the run also works against
Milvus Lite, which cannot serve the ordered pages the unbounded reads
use past it.

## Illustrative result (Milvus Lite, 20k items x 2k cats, dim=64)

Milvus Lite is an embedded dev server — treat these as shape, not
absolute performance. `example_infra/milvus.compose.yml` starts a real
standalone server to measure against.

```
1. ANN top-50 (no join, baseline)                p50=  44.6ms  rows=50
2. ANN top-50 JOIN cats (key pushdown)           p50=  25.1ms  rows=50
3. selective WHERE JOIN (predicate pushdown)     p50=  32.8ms  rows=89
4. GROUP BY over the join (broad read)           p50= 278.5ms  rows=10
```

The load-bearing observation is row counts, not milliseconds (Lite's
per-query jitter is large — that is why the p50 of 7 runs is reported):
query 2 reads at most 50 rows from `bench_cats` (not 2 000), query 3
reads ~89 rows from `bench_items` (not 20 000) — both scale with the
*result*, not the collection. Query 4 reads every matching row (~10k
here) by design; that is what a grouped aggregate means.
