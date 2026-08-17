# 0001 — The Milvus 3.0 boundary: what `milvusql` delegates and what it keeps

- **Status:** accepted (design only — no code in this document has been written yet)
- **Date:** 2026-08
- **Applies to:** `milvusql` 1.x and the 2.x line that will carry 3.0 support
- **Supersedes:** the "Milvus 3.0" paragraph in `README.md#compatibility`, which
  is now partly wrong (see [Docs that are now wrong](#docs-that-are-now-wrong))

## Why this exists

`milvusql` decides, statement by statement, what Milvus evaluates and what the
client evaluates. That line was drawn against Milvus 2.6, where the server sorts
nothing, groups nothing and aggregates nothing but `COUNT(*)`. Milvus 3.0 moves a
large part of that surface into the engine, which makes every client-side
operator here a candidate for deletion — and every one kept is one that has to
stay correct forever.

The line was therefore taken upstream as a design read rather than a feature
request, and the Milvus maintainers answered in detail. This document records
what they settled, what follows for this codebase, and what is still open. It is
the reference the 3.0 work is planned against; it is not itself an
implementation.

Maintainer positions below are paraphrased from that public discussion and are
attributed to "upstream" rather than to individuals.

## What upstream settled

### 1. Wire compatibility: progressive migration, not a port

The 2.6 client line works against a 3.0 server — the protocol evolves
additively, so existing 2.6 interfaces keep working. What a 2.6 client *cannot*
reach is the new 3.0 capability set: the first-class `TEXT` type, in-kernel
`GROUP BY`/`ORDER BY`/aggregates, `SearchAggregation`, and native cursor
iteration (`SearchIteratorV2Info`, `QueryIteratorCursor`).

**Consequence.** 3.0 support is not an all-or-nothing rewrite gated on a
dependency bump. Today's `pymilvus>=2.6.17,<3` build keeps working against a 3.0
server unchanged, and the 3.x client is adopted per capability, when there is
something specific to unlock. This removes the deadline pressure that the
original compatibility note assumed, and it means the two execution strategies
(client-side, server-side) must coexist in one codebase rather than replace each
other on a flag day.

### 2. The engine is absorbing the single-collection relational surface

3.0 already executes, in the kernel: `GROUP BY` with `COUNT`/`SUM`/`AVG`/`MIN`/
`MAX`, multi-field composite grouping, and `ORDER BY` (merge-sort across query
nodes, with `NULLS FIRST`/`NULLS LAST`). 3.1 adds predicate pushdown with
page-index and bloom-filter pruning, plus UDFs. The stated direction is that this
continues.

Upstream's explicit recommendation: **do not implement single-collection
`GROUP BY`/`ORDER BY`/aggregates client-side**; push them to the SDK
(`group_by_fields`, `order_by_fields`, `SearchAggregation`) and let the SQL layer
degenerate into SQL → SDK translation for those shapes. The operators the engine
will *not* take are the ones to keep: cross-collection joins, window functions,
CTEs, set operations.

### 3. Cross-collection `JOIN` is undecided, not declined

There is no JOIN structure anywhere in the 3.0 protos and nothing on the 3.0/3.1
roadmap, but the team has not made a final call either. Upstream's advice is to
bet on neither outcome: keep the plan-per-collection + client-side reduction as a
general-purpose gap-filler, behind an interface replaceable by server-side JOIN
plumbing without a rewrite.

Separately, the simpler shape — query one collection for a set of IDs, then
filter another collection by those IDs — is a pattern upstream may support more
directly. That is exactly what this codebase's equi-key pushdown already does
(`relational._plan_pushdown` / `_apply_pushdown`, lowered to `key in [...]`).

### 4. The DBAPI premise is confirmed

`pymilvus` has no PEP 249 implementation and no plans for one; its surface is
search/query-shaped (`MilvusClient`, ORM, iterators), not
connection/cursor/`fetchone`-shaped. The Spark DataSource v2 connector serves the
batch/analytics/backfill path, not the request path. A DBAPI for application
servers already on SQLAlchemy or Django is a real and currently unfilled gap.

One strategic caveat, from the same discussion: upstream does have plans for a
**PostgreSQL-compatible connector/interface**, with a deliberately simple initial
scope — no full relational engine, no complex JOINs. That is adjacent ground, not
the same ground, but it has two consequences worth acting on now:

- Where the SQL surface has a free choice, spell it the way PostgreSQL does. The
  dialect already does this in the places that matter (`<=>` from pgvector,
  `TEXT`), and it should stay that way so the two surfaces do not fork.
- Keep the execution backend behind `translate._common.Call` rather than letting
  `pymilvus` call shapes leak outward, so that "same SQL, different transport"
  stays a possible future rather than a rewrite.

### 5. `TEXT` is the engine's, and 16384 is not a boundary

Map `TEXT` straight onto the native 3.0 `TEXT` column, and full-text/BM25 onto
the engine's text-match capability. The 2.6 spelling this library uses —
analyzer-enabled `VARCHAR` plus an explicit generated sparse field — is a manual
workaround for an engine that did not yet own text; on 3.0 the engine natively
manages `TEXT` + BM25 (channel-local IDF) + text-match/analyzer, and replicating
the generated-field pipeline there is a leaky abstraction.

On paging: the 16384 ceiling is **not** a hard line. `topK` and
`maxQueryResultWindow` are server-configurable, a `largeTopK` path (default
1,000,000) exists since 2.6.14, and 3.0 has native cursor iteration
(`SearchIteratorV2Info`, `QueryIteratorCursor`). The home-grown primary-key
cursor should give way to the official iterator.

## The line, stated once

> Push single-collection filtering, sorting, aggregation, grouping and vector
> search into Milvus wherever Milvus has native semantics for them. Keep the SQL
> layer focused on compatibility, planning, and the relational constructs Milvus
> deliberately does not become a database for.

With one amendment this codebase has to add, because it supports servers that do
not have those native semantics: **the client-side operators are not deleted,
they become the floor.** Milvus Lite and 2.6 remain supported for the life of the
1.x line, so every operator delegated to a 3.0 engine must retain its client-side
implementation as the fallback. What changes is which path is *primary*, not
whether the other exists.

That amendment is the entire reason the next section is necessary.

## What this means for the code as it stands

| Today | Where | Verdict |
|---|---|---|
| `SUM`/`AVG`/`MIN`/`MAX`/`COUNT(col)` fetch every matching row and reduce in Python | `translate/ast_to_pymilvus.py:926`, `:965` | Delegate on 3.0; keep as floor |
| `COUNT(*)` is already server-side | `translate/ast_to_pymilvus.py:986` | Correct as-is — the existing precedent for delegation |
| Scalar `ORDER BY` sorts every matching row in Python, then applies `LIMIT` | `translate/ast_to_pymilvus.py:752` | Delegate on 3.0; keep as floor. Pin NULL ordering first |
| Any `GROUP BY` routes to the Polars engine, single-collection included | `translate/relational.py:253` | Route single-collection grouping back to the single-call path on 3.0 |
| `needs_relational_engine` decides purely on statement syntax | `translate/relational.py:253` | Must become capability-aware; this is the structural change |
| `TEXT` requires an explicit generated `SPARSEVEC` for full text | `translate/ast_to_pymilvus.py:248`, `:293` | Desugar now (2.6); native `TEXT` on 3.0 |
| `DEFAULT_QUERY_LIMIT = 16384` hardcoded as the ceiling | `translate/_common.py:40` | Wrong assumption even on 2.6 — the window is server-configurable |
| Hand-rolled primary-key cursor pages for unbounded reads | `translate/_common.py:408` | Replace with native iterators where available; keep for Lite |
| No paging on the `search` path at all | `translate/ast_to_pymilvus.py:1231` | `SearchIteratorV2` closes this |
| `ROW_NUMBER() OVER (PARTITION BY … ORDER BY distance)` always falls to Polars | `translate/relational.py:299` | Maps to Grouping Search — available on 2.6, no client bump needed |
| Cross-collection planning and Polars execution are already split | `translate/relational.py` `_plan_*` vs `_evaluate`/`_join` | Make the split load-bearing and documented; it is the replaceable interface |

Two findings from that review are worth separating out, because they are defects
today rather than 3.0 work:

**The hardcoded ceiling is an honesty risk.** `unbounded_query_call` sends
`limit=16384` and treats "fewer than 16384 rows came back" as "the result is
complete" (`translate/_common.py:408`). If an operator has configured
`maxQueryResultWindow` *below* 16384, that inference is unsound and the read
truncates silently — precisely the failure mode the project's ground rules forbid.
Whether the server errors instead of capping needs verifying against a real
server before deciding the fix; either way the constant should come from the
server, not from source.

**Two sort paths, two NULL semantics.** The client-side sort orders NULLs last on
`ASC` and first on `DESC` (`translate/ast_to_pymilvus.py:772`). 3.0 supports
explicit `NULLS FIRST`/`NULLS LAST`. Until both paths are pinned to the same
answer by tests, delegating `ORDER BY` changes results for some queries — which
means the tests come first, not after.

## The capability layer

Everything in section 2 needs the same missing piece: the translator must know
what the server can do. There is no version or feature detection anywhere in the
codebase today.

The constraint is `CONTRIBUTING.md`'s rule that translation code performs no I/O.
The precedent for satisfying it already exists: `_build_create_table` receives the
client and calls `create_schema()` on it (`translate/ast_to_pymilvus.py:293`), and
`unbounded_query_call` obtains runtime facts by *chaining* a `describe_collection`
`Call` rather than issuing one. Capability detection should follow the first
pattern, not the second — a probe per statement is not acceptable.

Proposed shape:

- A frozen `Capabilities` struct — server-side ordering, grouping, the set of
  aggregate *shapes* (not just function names) computed in the kernel, native
  `TEXT`, native iterators, grouping search, the real result window.
- Resolved **once per `Connection`**, lazily, from one `get_server_version()`
  probe plus the installed `pymilvus` major, and cached on the connection
  alongside `_loaded_collections`.
- Overridable by an explicit connection parameter, for operators who know their
  deployment and do not want the probe, and to make the matrix testable without
  a server of each version.
- Threaded into `build_call(...)` and `needs_relational_engine(...)` as an
  argument, exactly as `client` already is.
- Defaulting to the **conservative floor** on any uncertainty. An unknown server
  gets today's behaviour, which is correct everywhere and merely slower where the
  engine could have helped.

Capabilities must be keyed on shape, not on function name. In-kernel aggregates
are documented as `COUNT`/`SUM`/`AVG`/`MIN`/`MAX`; `COUNT(DISTINCT x)` is a
different question and, until confirmed, stays client-side. The struct should be
able to say "SUM yes, SUM DISTINCT no".

**The cost, stated plainly.** This turns one dispatch table into a matrix of
statement shape × capability set, and roughly doubles the behavioural test
surface for every delegated operator. That is the price of supporting Lite, 2.6
and 3.0 from one codebase, and it is worth paying only because the alternative —
maintaining client-side implementations of operators the engine does better,
forever — is worse. The mitigation is that the matrix has exactly two rows worth
testing exhaustively (floor, and full 3.0), parametrized from the same test
bodies, plus a 3.0 container in the integration tier.

## Work items

Ordered by dependency, not by value. "Client" is the `pymilvus` line each item
requires.

| # | Item | Client | Notes |
|---|---|---|---|
| 1 | Pin NULL ordering, empty-set aggregate and `DISTINCT` semantics with tests that both paths must satisfy | 2.6 | Gates #7. Pure test work on today's behaviour |
| 2 | Read the result window from the server instead of hardcoding 16384; verify the below-16384 truncation case | 2.6 | Correctness fix, independent of 3.0 |
| 3 | `TEXT` alone implies the full-text pipeline; the explicit generated-`SPARSEVEC` spelling becomes opt-in for analyzer configuration | 2.6 | Aligns the 2.6 surface with what 3.0 will do natively, so the DDL does not change again later |
| 4 | Compile top-k-per-group to Grouping Search | 2.6 | Verify semantics first — see below |
| 5 | The capability layer | 2.6 | Gates #6, #7, #8 |
| 6 | Native `TEXT`; `DESCRIBE` round-trips it | 3.x | Introspection and Django migrations must agree on the spelling |
| 7 | Delegate single-collection `ORDER BY`, `GROUP BY` and aggregates | 3.x | The bulk of the win. Needs #1 and #5 |
| 8 | Native `QueryIteratorCursor` / `SearchIteratorV2`; keep the hand-rolled loop for Lite | 3.x | Also closes the missing `search` paging. Async parity is an open question |
| 9 | Document the plan IR as the replaceable cross-collection interface; keep ID-set pushdown a distinct plan node | 2.6 | Cheap insurance against either outcome of section 3 |
| 10 | Update the compatibility matrix and the pushdown table | — | See below |

### On item 4, and why it is not free

Upstream mapped `ROW_NUMBER() OVER (PARTITION BY category ORDER BY distance)`
with `row_number <= K` to Grouping Search, and that is the right mapping for the
ANN case — facets are bucket statistics, grouping search changes which hits come
back per group.

The caveat is a semantic mismatch that has to be resolved before this ships. SQL's
window formulation asks for *every* group, with the top K hits inside each.
Grouping Search bounds the number of groups by the search `limit` and the hits
per group by `group_size`. Those are the same answer only when the group
cardinality fits under the limit. Under the project's honesty rule, returning
"the top N groups" for a query that asked for all of them is exactly the kind of
quietly-reinterpreted result that is not allowed.

So the translation is faithful only where the statement itself bounds the group
count, or where cardinality is known to fit. Everything else keeps the Polars
path. The precise behaviour of `limit`, `group_size` and `strict_group_size`
needs confirming against a real server before the recognizer is written — this
document does not assume it.

## What does not change

- **The client-side relational engine stays.** Section 3 makes it a permanent
  component, not a stopgap awaiting deletion, and even a future server-side JOIN
  would not remove it while Lite and 2.6 are supported.
- **The honesty rule stands, and gets harder.** Two execution paths for one
  operator means two chances to disagree. A capability that cannot be proven to
  produce the same answer as the floor does not get delegated.
- **One dispatch table, two thin call sites.** The capability struct is an
  argument to the builders; it must not become a second dispatch path, and it
  must not tempt either cursor into doing translation work.
- **The premise.** Section 4 confirms the DBAPI gap is real. Nothing here
  suggests narrowing scope.

## Open questions for upstream

1. Does the 3.x async client have `query_iterator`/`search_iterator` parity? The
   2.6 `AsyncMilvusClient` has neither, which is why the page loop here is
   hand-rolled from plain `query` calls in the first place. If 3.x still lacks
   them, item 8 applies to the sync cursor only and the two cursors diverge —
   which this codebase's D1/D12 rule specifically exists to prevent.
2. Are in-kernel aggregates `DISTINCT`-aware? Is `COUNT(DISTINCT x)` expressible?
3. Do in-kernel `GROUP BY`/`ORDER BY` apply to `search` results as well as
   `query`, or to `query` only? The answer decides whether a filtered ANN search
   with grouping is one RPC or two.
4. Is there a server-side `HAVING` equivalent, or does grouped filtering
   necessarily come back to the client?
5. Grouping Search: does `limit` bound groups or rows, and what exactly does
   `strict_group_size` guarantee? (Item 4 above.)
6. Does a 3.0 server report `maxQueryResultWindow`/`topK` to the client, or does
   a client have to discover the window by hitting it? (Item 2 above.)
7. The PostgreSQL-compatible connector: is it a wire-protocol server or a dialect
   surface, and roughly when? If it is wire-protocol, a DBAPI on top of it is a
   different and much smaller piece of work than a DBAPI on top of the SDK.

## Docs that are now wrong

`README.md#compatibility` currently states that `milvusql` 1.x against a 3.0
server is **"Not supported"**, and that whether the 16384-row ceiling still
applies on 3.0 is "unverified". Section 1 and section 5 answer both:

- The 2.6 client line does work against a 3.0 server; what it cannot do is reach
  3.0-only capabilities. The row should say so rather than reading as a hard
  incompatibility, with the caveat that this is upstream's statement and is not
  yet covered by this repository's CI.
- The ceiling is server-configurable and not a fundamental boundary, on 2.6 as
  well as 3.0.

The pushdown table in `README.md#join-group-by-and-subqueries` will need
revisiting alongside item 7, since its "evaluated client-side" column becomes
conditional on the server rather than absolute.

Neither is edited here: this document is the design record, and the README
changes belong with the work they describe.
