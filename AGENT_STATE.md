# Agent State — milvusql / discussion #52546

Journal for the periodic maintenance agent that tracks the commitments made
in [milvus-io/milvus#52546](https://github.com/milvus-io/milvus/discussions/52546).
One run inspects the repository, compares it against the list below, takes the
single smallest item that still moves the project, and opens one PR.

```
Last run:                  2026-09-14
Discussion last comment:   2026-08-17 (Neko1313's closing reply; nothing new since)
Milvus latest release:     3.0.1 (2026-09-09) — no 3.1, so no new obligations
Repository HEAD assessed:  e3a2a7e (main)
```

## Checklist

| # | Commitment | Status | Where it stands |
|---|---|---|---|
| 1 | Progressive-migration 2.6→3.0 documented as a README compatibility table | `done` | `README.md` "Compatibility", landed in `e3a2a7e` |
| 2 | Server-capability layer instead of a hardcoded "always client-side" | `in_progress` | `src/milvusql/capabilities.py` + `Connection.capabilities()`; call sites still constant |
| 3 | Cross-collection ID-set/equi-join behind a replaceable interface | `todo` | Resolution is inline in `translate/relational.py`; no protocol/ABC yet |
| 4 | Native `TEXT` on 3.0, generated-`SPARSEVEC` as a deprecated 2.6 path | `todo` | Flag `native_text_field` exists; the dialect branch and the deprecation warning do not |
| 5 | Cursor pagination around an iterator abstraction, PK-cursor as legacy | `todo` | PK-cursor page loop in `translate/_common.py`; flags `search_iterator_cursor`/`query_iterator_cursor` exist, nothing switches on them |
| 6 | `ROW_NUMBER() OVER (PARTITION BY ...)` + `rn <= K` → Grouping Search | `todo` | Recognized as a window function and evaluated client-side (`translate/relational.py`); no `group_by_field`/`group_size` mapping |
| 7 | PostgreSQL-native alternatives alongside the MySQL-style surface | `todo` | Grammar lives in `Callix-Tools/sqlglot-milvus`, so this spans two repositories |
| 8 | Docs integration page ("Milvus from SQLAlchemy / from Django") | `todo` | No `docs/` here; the published docs live in `Callix-Tools/milvusql-docs` |
| 9 | Exact query aggregation vs approximate search aggregation kept distinct | `todo` | Attempted in PR #14, closed unmerged |
| 10 | 16384 no longer presented as a fundamental SQL-layer boundary | `in_progress` | Code frames it as Milvus's per-call ceiling and pages past it; `per_call_row_ceiling` now encodes it as a server property. README still hedges ("whether it applies on 3.0 is unverified") where the discussion answered it |

### Open PRs from earlier runs

| PR | Item | State |
|---|---|---|
| [#17](https://github.com/Callix-Tools/milvusql/pull/17) | 9 (adjacent: reject a computed SELECT-list expression) | `blocked: awaiting review` — opened 2026-09-07, no review after 7 days, mergeable |

Closed unmerged, so their work is not in `main`: #11, #12, #13 (item 6), #14 (item 9),
#15, #16.

## Run log

### 2026-09-14

- **Chosen:** item 2 — the server-capability layer, as the table plus its probe,
  with no call site rewired.
- **Rationale:** it is the item every other deferred one waits on (4, 5 and 6 are
  each "do X where the server supports it"), and it is the only unattempted item
  that can land as a self-contained, testable unit. Splitting the wiring out keeps
  this diff additive: no existing behaviour changes.
- **Result:** `src/milvusql/capabilities.py`, `Connection.capabilities()` and
  `AsyncConnection.capabilities()`, 38 unit tests. Core suite 287 → 325 passed;
  the two sub-packages unchanged at 86 and 57. `ruff`, `ty` and `bandit` clean.
- **Diff size:** ~430 lines, 6 files (of which two are one-line touches).

Two things were confirmed against the installed `pymilvus` 2.6.17 rather than
assumed, and both change what later runs should do:

- `group_by_field`, `group_size` and `strict_group_size` already reach the wire
  from the 2.6 client (`pymilvus/client/prepare.py` folds them into
  `search_params`). **Item 6 is not gated on Milvus 3.0** — Grouping Search is
  available on the pairing this release already supports.
- `group_by_fields` (the plural, kernel-`GROUP BY` spelling) appears nowhere in
  2.6.17, and neither does `QueryIteratorCursor`. `SearchIteratorV2` does. So
  items 4 and 5's query-side halves are genuinely blocked on a pymilvus 3 client,
  while the search-side cursor is not.

Milvus Lite reports its version as `milvus_lite-3.2.0` (confirmed against a real
embedded connection). That trailing number is the `milvus-lite` package version,
not a Milvus release line: any future version check must special-case it or it
will read the least capable deployment as the most capable one.

**Notes for later, deliberately not in this diff:** the README hedge on the
16384 ceiling (item 10) is now contradicted by the discussion and should be
rewritten when item 5 lands; item 8's natural home is the `milvusql-docs`
repository, not a new `docs/` here, so it needs a decision before work starts.
