# Agent State — milvusql / discussion #52546

Journal for the periodic maintenance pass that works the commitments made
in [milvus-io/milvus#52546](https://github.com/milvus-io/milvus/discussions/52546).
One pass, one item, one pull request.

Last run: 2026-09-04
Discussion last checked: 2026-09-04 — newest comment still 2026-08-17
(five comments: Neko1313 ×3, `yhmo`, `xiaofan-luan`; no new obligations)
Milvus release checked: not re-checked this pass (upstream repo is outside
this session's GitHub scope; the discussion page carried no 3.1 news)

## Checklist

| # | Commitment | Status | PR | Checked |
|---|---|---|---|---|
| 1 | Progressive migration 2.6→3.0 documented as a README compatibility table | done | — (main `e3a2a7e`) | 2026-09-04 |
| 2 | Server-capability layer: single-collection `GROUP BY`/`ORDER BY`/aggregates delegated where the server supports them, Polars only as fallback | todo | — | 2026-09-04 |
| 3 | Cross-collection ID-set/equi-join pattern behind a replaceable interface | todo | — | 2026-09-04 |
| 4 | `TEXT` mapped natively on 3.0; generated `SPARSEVEC` spelling becomes the legacy 2.6 path with a deprecation warning | todo | — | 2026-09-04 |
| 5 | Cursor pagination built on an iterator abstraction (`SearchIteratorV2Info`/`QueryIteratorCursor` on 3.x), PK cursor as the 2.6 legacy path | todo | — | 2026-09-04 |
| 6 | `ROW_NUMBER() OVER (PARTITION BY … ORDER BY <vector>)` + `rn <= K` compiled to Grouping Search | todo | [#13](https://github.com/Callix-Tools/milvusql/pull/13) (closed unmerged) | 2026-09-04 |
| 7 | PostgreSQL-native alternatives alongside the MySQL-style constructs (`MATCH … AGAINST`, `AUTO_INCREMENT`, `SHOW TABLES`) | todo | — | 2026-09-04 |
| 8 | Docs integration page ("use Milvus from SQLAlchemy/Django") drafted, ready to offer upstream | todo | — | 2026-09-04 |
| 9 | Aggregation accuracy: an exact SQL aggregate must not silently compile onto an approximate ANN search path | in_progress | this run | 2026-09-04 |
| 10 | 16384 no longer presented as a fundamental ceiling of the SQL layer | done | — (main `e3a2a7e`) | 2026-09-04 |

### Notes on the current statuses

- **#1, #10** — the README carries the compatibility table, and the row
  ceiling is described as a per-call limit the reader pages past
  (primary-key cursor), not as a boundary of the SQL layer. Milvus Lite is
  named as the one server that cannot serve ordered pages.
- **#2** — nothing in `src/` reads a client or server version to decide
  where work runs; the Polars path is chosen purely by statement shape
  (`relational.needs_relational_engine`). This is the item that unblocks
  #5 and the delegation half of #6.
- **#3** — the equi-join key pushdown exists and works
  (`relational._plan_joins` → `key in [...]` on the next scan), but as
  inline planner logic, not behind a protocol a server-side JOIN could
  later replace.
- **#4** — `TEXT` compiles to an analyzer-enabled `VARCHAR`
  (`ast_to_pymilvus._map_datatype`); there is no native-`TEXT` branch and
  no deprecation warning on the generated `SPARSEVEC` spelling yet.
- **#6** — recognition of the pattern was proposed in PR #13 and closed
  without merging on 2026-09-04. Not re-opened by this pass: a closed
  proposal is a decision to leave to the owner, not to re-submit. Waiting
  on direction before spending another pass here.

## Run log

### 2026-09-04

- **Chosen:** #9 — the exact-vs-approximate aggregation guard.
- **Rationale:** #6 had just been declined, #2/#3/#4/#5 each need an
  architectural seam larger than one pass, and the inventory turned up a
  live silent mistranslation squarely inside #9's wording: an aggregate
  carrying a search `ORDER BY` had the search dropped.
- **Found:** `SELECT COUNT(*) FROM items ORDER BY embedding <=> :q LIMIT 10`
  compiled to `query(collection_name='items', output_fields=['count(*)'])` —
  the whole collection's count returned for a question asked about ten rows,
  with no error. `SELECT AVG(price) FROM items ORDER BY … LIMIT 10` likewise
  averaged over the collection, and `BM25_SCORE` ordering behaved the same.
- **Result:** `_build_aggregate` now refuses the shape with a
  `NotSupportedError` naming the exactness difference and the two-step
  spelling that works. README documents both. Verified by direct execution
  before and after.
- **Diff size:** ~180 lines, 3 files (`ast_to_pymilvus.py`,
  `select_aggregate/error.py`, `README.md`) plus this journal.
- **Left open on #9:** the grouped path (`GROUP BY … ORDER BY <vector>
  LIMIT k`) does apply the search and then reduces its hits — exact over an
  approximate candidate set. That is documented now, not flagged at runtime;
  whether it should warn is a judgement call for the owner.

## Notes for future passes (not obligations)

- A distance operator outside `ORDER BY` — in a projection
  (`SELECT id, embedding <=> :q AS d FROM items GROUP BY id`) or a `HAVING`
  — plans as a plain scan that fetches the whole embedding column, and then
  fails during evaluation with a bare `ValueError: Unsupported expression
  type CosineDistance` raised by sqlglot's own generator
  (`sqlglot/generator.py`), not by `relational.py`'s
  `NotSupportedError`. Two problems in one: an exception outside the PEP 249
  hierarchy reaching the caller, and every vector in the collection read
  before anyone notices. Verified directly on 2026-09-04. Small,
  self-contained, adjacent to #9 — a good candidate for a future pass, and
  deliberately not folded into this one's diff.
- `CHANGELOG.md` is generated by git-cliff from conventional commits — never
  edit it by hand.
