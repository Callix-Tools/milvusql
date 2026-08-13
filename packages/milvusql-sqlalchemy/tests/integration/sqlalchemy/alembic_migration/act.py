"""Integration coverage: Alembic against a real Milvus server (via
``testcontainers``) surfaces a clear, explicit error instead of either
silently working or failing with an opaque low-level one.

This backend does *not* make Alembic work end to end -- confirmed
directly, and left that way on purpose: Alembic's own
``alembic_version`` bookkeeping table (``version_num VARCHAR(32)
PRIMARY KEY``, no vector column of its own) can never be created,
because Milvus refuses ``CREATE TABLE``/``create_collection()``
outright for a schema with zero vector fields. That's a real,
documented Milvus limitation, not a gap in this dialect worth papering
over with a hidden pad column (an earlier version of this coverage did
exactly that, and it made *inserting into* such a table fail instead,
which is worse -- a `CREATE TABLE`-time failure with a clear message
is the more honest outcome). What this dialect *does* own is turning
Alembic's own generic ``KeyError`` (an unregistered dialect name, see
``alembic_impl.py``) and Milvus's own opaque gRPC rejection into one
clear, explicit ``NotSupportedError`` raised from
``ast_to_pymilvus._build_create_table`` client-side, before the RPC
even goes out -- this test proves that's genuinely what a real
``alembic upgrade`` run surfaces.

``registers-with-alembic`` unit coverage (``alembic_impl.py``'s
``MilvusImpl``) and the ``NotSupportedError`` unit coverage
(``tests/unit/translate/create_table/error.py``) both matter on their
own -- this integration test is what proves they compose correctly
against a real server, not just individually."""

from __future__ import annotations

import textwrap

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.exc import NotSupportedError

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


    def downgrade():
        op.drop_table("widgets")
    '''
)


@pytest.fixture
def alembic_config(tmp_path, db_uri):
    """A real, on-disk Alembic project (``env.py`` + one versioned
    migration script, itself a perfectly valid ``CREATE TABLE`` with a
    real vector column) pointed at this worker's real Milvus
    server/database. The migration script never actually runs -- the
    failure this test expects happens earlier, the moment
    ``MigrationContext.configure()`` tries to create
    ``alembic_version`` itself."""
    versions_dir = tmp_path / "versions"
    versions_dir.mkdir()
    (versions_dir / "0001_create_widgets.py").write_text(_MIGRATION_SCRIPT)
    (tmp_path / "env.py").write_text(_ENV_PY)

    cfg = Config()
    cfg.set_main_option("script_location", str(tmp_path))
    cfg.set_main_option("sqlalchemy.url", db_uri)
    return cfg


def test_upgrade_fails_with_a_clear_error_not_an_opaque_one(alembic_config):
    with pytest.raises(NotSupportedError, match="VECTOR"):
        command.upgrade(alembic_config, "head")
