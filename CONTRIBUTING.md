# Contributing to milvusql

Thanks for considering a contribution! This repository is a `uv`
workspace: the core DBAPI (`src/milvusql`) plus two packages built on
it (`packages/milvusql-sqlalchemy`, `packages/milvusql-django`).

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and
[task](https://taskfile.dev/):

```bash
task install           # uv sync --all-groups --all-packages
task lint              # ruff + ty + bandit, core + all packages
task tests             # unit + integration (integration needs Docker)
```

Integration tests start one real Milvus standalone container per test
session via `testcontainers` — Docker is required for them; a plain
`uv run pytest tests/unit` run never touches it. To point them at an
already-running server instead, set `MILVUS_TEST_URI=http://host:port`.

## Ground rules

- **Honest errors over plausible wrong answers.** A query that cannot
  be answered correctly raises with an actionable message; it never
  returns a truncated or reinterpreted result. Every feature in this
  codebase follows that rule — new ones must too.
- **One dispatch table, two thin call sites.** Translation code
  (`translate/`) never performs I/O; `dbapi/cursor.py` (sync) and
  `aio.py` (async) are the only two places a client method is invoked.
  A feature that works on one cursor must work identically on the
  other.
- **Tests mirror the tree.** `tests/{unit,integration}/<entity>/<action>/`
  holds `act.py` (happy path) and `error.py` (error/edge cases). Unit
  tests parse real MilvusQL through the real grammar — no hand-built
  AST nodes.
- **Conventional commits.** Release automation (git-cliff) derives
  versions and the changelog from commit messages: `feat:`, `fix:`,
  `docs:`, `test:`, `chore:` — scoped where useful
  (`feat(sqlalchemy): ...`).

## Pull requests

1. Fork, branch, make the change with tests.
2. `task lint && task tests` locally (at least the unit tier).
3. Open the PR against `main` with a conventional-commit title.

## Releases

Maintainers run the `Release` workflow (GitHub Actions,
`workflow_dispatch`): it derives the next version from conventional
commits, bumps every workspace package in lockstep, regenerates the
changelog, tags, publishes to PyPI and cuts a GitHub release.
