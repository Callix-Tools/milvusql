# Agent State — milvusql / discussion #52546

Journal kept by the periodic maintenance agent between runs. It exists so a
run can tell, before it starts, what has already been done, what is waiting
on review, and what was deliberately deferred — the first two runs that had
no journal re-opened the same change twice (see the run log).

- Last run: 2026-09-07
- Discussion last checked: **not verified this run** — the session's GitHub
  access is scoped to `Callix-Tools/milvusql` and `Callix-Tools/sqlglot-milvus`,
  so `milvus-io/milvus` discussion #52546 could not be read. The checklist
  below is carried forward from the September 2026 snapshot and confirmed
  against the code, not against new upstream comments.
- Milvus release checked: **not verified this run** — same scope limitation.
  Items 2, 5 and 6 depend on what 3.0/3.1 actually ships.

## Checklist

Status reflects `main` at `e3a2a7e`. "Evidence" is where the current
behaviour lives, so the next run can re-check it without re-reading the
whole tree.

| # | Item | Status | Evidence / PR | Updated |
|---|---|---|---|---|
| 1 | Progressive-migration 2.6→3.0 as a README compatibility table | done | `README.md` "Compatibility"; correction in flight as [#15](https://github.com/Callix-Tools/milvusql/pull/15) | 2026-09-07 |
| 2 | Server-capability layer (resolve to `query(group_by_fields=…)`/`search` where the server supports it, Polars otherwise) | todo | no version or capability detection anywhere in `src/` or `packages/` | 2026-09-07 |
| 3 | Cross-collection ID-set/equi-join behind a replaceable interface | todo | `translate/relational.py` `_plan_join` / `_apply_semi_join` are module-level functions, no protocol or ABC | 2026-09-07 |
| 4 | Native `TEXT` on 3.0; generated `SPARSEVEC` spelling as a deprecated 2.6 path | todo | `translate/ast_to_pymilvus.py:189` maps `TEXT` onto analyzer-enabled `VARCHAR` unconditionally; no deprecation warning in `src/` | 2026-09-07 |
| 5 | Cursor pagination on an iterator abstraction, PK-cursor as the 2.6 legacy path | todo | `translate/_common.py:393` builds PK-cursor pages from plain `query` calls; no `SearchIteratorV2` / `QueryIteratorCursor` use | 2026-09-07 |
| 6 | `ROW_NUMBER() OVER (PARTITION BY … ORDER BY … <=> :q)` + `rn <= K` → Grouping Search | todo | window functions route to the Polars ranking path, `translate/relational.py:2284`; [#13](https://github.com/Callix-Tools/milvusql/pull/13) attempted this and was **closed unmerged** | 2026-09-07 |
| 7 | PostgreSQL-native alternatives alongside the MySQL-style full-text/DDL spellings | todo | `sqlglot-milvus` `dialect.py` has `MATCH … AGAINST`, `AUTO_INCREMENT`, `SHOW TABLES`; no PG-equivalent branches | 2026-09-07 |
| 8 | Docs integration page ("Milvus from SQLAlchemy / from Django") | todo | no `docs/` directory in this repo | 2026-09-07 |
| 9 | Exact `GROUP BY … COUNT(*)` must not silently map to approximate search aggregation | in_progress | [#14](https://github.com/Callix-Tools/milvusql/pull/14) open, CI green | 2026-09-07 |
| 10 | 16384 no longer stated as a fundamental SQL-layer boundary | done | README frames it as a per-call ceiling that milvusql pages past, and marks 3.0 applicability unverified; `DEFAULT_QUERY_LIMIT = 16384` in `translate/_common.py:40` is a per-call request limit, not a documented hard ceiling | 2026-09-07 |

## Open work waiting on a human

| PR | Item | Opened | State |
|---|---|---|---|
| [#14](https://github.com/Callix-Tools/milvusql/pull/14) | 9 | 2026-09-04 | draft, CI green, awaiting review |
| [#15](https://github.com/Callix-Tools/milvusql/pull/15) | 1 | 2026-09-05 | draft, awaiting review |

## Notes for future runs

- **Read this file and `gh pr list --state all` before choosing work.** PR #15
  re-does PR #12, which was closed unmerged four days earlier; nothing in the
  repo recorded that #12 had already been tried.
- Several agent PRs were closed without merging (#11, #12, #13). Treat a
  closed PR as a signal that the approach was not wanted, not as unfinished
  work to retry unchanged.
- Item 2 (capability layer) unblocks 5 and 6. Items 4, 5 and 6 all need to
  know what the connected server and client actually support, so 2 is the
  natural next unit of work once the queue clears.
- Deferred, not committed to (noted so it is not smuggled into an unrelated
  diff): the join planning in `relational.py` would need extracting before
  item 3 can be done as a small change.

## Run log

### 2026-09-07
- Chosen: create this journal. No new checklist item started — #14 and #15
  were opened on 2026-09-04 and 2026-09-05 and are still unreviewed, and the
  run rule is one item at a time while earlier work is unmerged.
- Rationale: the journal is the first source of truth the run procedure asks
  for and it did not exist, which had already cost one duplicate PR.
- Verified before writing: `uv run ruff check --no-fix src tests` clean,
  `uv run pytest tests/unit` 287 passed.
- Not verified: discussion #52546 and Milvus release status — out of the
  session's repository scope.
- Diff size: 1 file, docs only.
