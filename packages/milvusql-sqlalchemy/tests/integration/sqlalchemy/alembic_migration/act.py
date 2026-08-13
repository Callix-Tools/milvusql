"""Integration coverage: real Alembic migrations against a real Milvus
server (via ``testcontainers``), driven through the exact same
``alembic upgrade``/``downgrade`` workflow a real project uses --
``alembic.command.upgrade()``/``downgrade()`` running a generated
``env.py`` and a versioned migration script from disk, not just
``Operations`` calls made directly against a bare
``MigrationContext``.

This is what actually proves Alembic works against this dialect end
to end: ``command.upgrade()`` first creates Alembic's own
``alembic_version`` bookkeeping table (``version_num VARCHAR(32)
PRIMARY KEY``, no vector column of its own) through
``MigrationContext.configure()`` -- Milvus refuses ``CREATE TABLE``
outright for a schema with zero vector fields, confirmed directly
against a real (non-Lite) server, which is exactly what
``milvusql_sqlalchemy.padding``'s hidden pad-vector column (spliced
into ``CREATE TABLE`` text by ``compiler.py``'s ``visit_create_table``,
indexed and loaded by ``dialect.py``'s ``after_create`` event
listener) exists to make possible -- then runs the migration script's
own ``upgrade()``, which creates a real table with a genuine vector
column via ``op.create_table``/``op.add_column``, exercising the same
``MilvusDDLCompiler`` path as ``metadata.create_all()`` elsewhere in
this suite.

``transactional_ddl=False`` in the generated ``env.py`` is deliberate,
not incidental: Milvus has no multi-statement rollback (D7, the same
reasoning ``MilvusDialect.do_rollback``'s own module docstring gives)
-- letting Alembic wrap each migration in a transaction it can never
actually roll back would be lying about a guarantee this backend
can't keep."""

from __future__ import annotations

import textwrap

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

pytestmark = [pytest.mark.integration, pytest.mark.sqlalchemy]

_ENV_PY = textwrap.dedent(
    """
    from alembic import context
    from sqlalchemy import create_engine

    config = context.config
    engine = create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=None,
            transactional_ddl=False,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
    """
)

_MIGRATION_SCRIPT = textwrap.dedent(
    '''
    """create widgets"""
    from alembic import op
    import sqlalchemy as sa
    from milvusql_sqlalchemy.types import VECTOR

    revision = "0001"
    down_revision = None
    branch_labels = None
    depends_on = None


    def upgrade():
        op.create_table(
            "widgets",
            sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(64)),
            sa.Column("embedding", VECTOR(4)),
            milvusql_shards=1,
        )
        op.add_column(
            "widgets", sa.Column("tag", sa.String(16), nullable=True)
        )


    def downgrade():
        op.drop_table("widgets")
    '''
)


@pytest.fixture
def alembic_config(tmp_path, db_uri):
    """A real, on-disk Alembic project (``env.py`` + one versioned
    migration script) pointed at this worker's real Milvus
    server/database -- built fresh per test so ``pytest-randomly`` can
    reorder freely without one test's ``alembic_version``/``widgets``
    state leaking into another's (``_milvus_worker_cleanup``, pulled
    in transitively through ``db_uri``, drops every collection --
    ``alembic_version`` included -- after each test)."""
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    (versions_dir / "0001_create_widgets.py").write_text(_MIGRATION_SCRIPT)
    (tmp_path / "env.py").write_text(_ENV_PY)

    cfg = Config()
    cfg.set_main_option("script_location", str(tmp_path))
    cfg.set_main_option("sqlalchemy.url", db_uri)
    return cfg


def test_upgrade_head_creates_the_table_and_records_the_revision(
    alembic_config, engine
):
    command.upgrade(alembic_config, "head")
    insp = inspect(engine)
    table_names = insp.get_table_names()
    assert "widgets" in table_names
    assert "alembic_version" in table_names

    columns = {col["name"] for col in insp.get_columns("widgets")}
    assert {"id", "name", "embedding", "tag"} <= columns

    with engine.connect() as conn:
        (version_num,) = conn.exec_driver_sql(
            'SELECT version_num FROM "alembic_version"'
        ).fetchone()
    assert version_num == "0001"


def test_downgrade_to_base_drops_the_table(alembic_config, engine):
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")
    insp = inspect(engine)
    assert "widgets" not in insp.get_table_names()
