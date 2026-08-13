"""``DatabaseSchemaEditor`` -- D11's flagged spike, and the honest
scope boundary of this first cut.

Deliberately does **not** inherit Django's base ``table_sql``/
``column_sql``/``_alter_field`` machinery: that machinery is built for
databases with full ``ALTER TABLE`` and deferred FK constraint SQL,
neither of which Milvus has. Milvus's real DDL surface (Track A's own
grammar) is much smaller -- ``CREATE TABLE``, ``ADD FIELD``, nothing
else -- so this hand-writes the small column-list builder it actually
needs instead of fighting the generic one.

**What works**: ``CreateModel`` (a model with scalar fields + one or
more ``VectorField``s), ``AddField`` -- against a real Milvus server.
Milvus Lite's gRPC server does not implement ``AddCollectionField``
(confirmed directly: a raw ``grpc.RpcError`` with ``UNIMPLEMENTED``,
not even a ``MilvusException``), so ``AddField`` still raises against
the embedded server this workspace's integration suite runs on -- see
``test_add_field_raises_not_supported_against_a_real_collection`` --
even though ``milvusql`` core's ``ALTER TABLE`` dispatch now wires it
through correctly. **What deliberately raises**: ``RemoveField``,
``AlterField`` -- Milvus cannot do either (same reasoning as
``sqlglot-milvus``'s own ``ALTER TABLE`` rejection), so this fails
loudly at migration time rather than silently producing a migration
that doesn't match reality.

**What this does NOT do**: automatically create a vector index or
``LOAD`` the collection after ``CreateModel`` -- Milvus requires an
index before a collection is searchable, but index method/metric are a
query-shape decision (HNSW vs. IVF, COSINE vs. L2), not something a
generic schema migration should guess. Call
``milvusql_django.schema.create_index_and_load`` explicitly (in a data
migration, or app startup) once the model is defined. This is the
single biggest open item this package leaves for later, flagged here
rather than papered over with a guessed default.
"""

from __future__ import annotations

import typing as t

from django.db.backends.base.schema import BaseDatabaseSchemaEditor

from milvusql_django.fields import VectorField

if t.TYPE_CHECKING:
    from django.db.models import Field

#: Milvus refuses `CREATE TABLE` outright for a schema with zero vector
#: fields (confirmed directly: `code=6, "schema has no vector field
#: (FLOAT_VECTOR or SPARSE_FLOAT_VECTOR)"`) -- a real, hard platform
#: requirement, not a choice this backend gets to skip. Django's own
#: internal bookkeeping models are exactly this shape: never a vector
#: field among them, sharpest example being
#: `django.db.migrations.recorder.MigrationRecorder.Migration`, which
#: `migrate` itself creates before a single user migration runs.
#: `create_model` splices in one hidden, always-populated `VECTOR(2)`
#: column for a model that has none, named unguessably enough that it
#: can never collide with a real field, and Django's own ORM never
#: sees or selects it -- only `CursorWrapper`'s `INSERT` padding
#: (`base.py`) and this module ever touch it. Dimension 2, not 1: a
#: real (non-Lite) Milvus server enforces its own documented minimum
#: vector dimension of 2 and rejects `dim=1` with `code=1100,
#: "invalid dimension: 1. should be in range 2 ~ 32768"` -- Milvus
#: Lite silently accepted a 1-dimensional vector with no such
#: validation, which is what let this go unnoticed until tested
#: against a real server.
PAD_VECTOR_FIELD = "_milvusql_pad_vector"
PAD_VECTOR_VALUE: list[float] = [0.0, 0.0]


def _needs_pad_vector(model: type[t.Any]) -> bool:
    return not any(
        isinstance(field, VectorField) for field in model._meta.local_fields
    )


class DatabaseSchemaEditor(BaseDatabaseSchemaEditor):
    sql_create_table = "CREATE TABLE %(table)s (%(definition)s)"
    sql_delete_table = "DROP TABLE %(table)s"

    def _column_definition(self, model: type[t.Any], field: Field) -> str:
        db_type = field.db_type(self.connection)
        parts = [self.quote_name(field.column), db_type]
        if field.primary_key:
            parts.append("PRIMARY KEY")
            if field is model._meta.auto_field:
                parts.append("AUTO_INCREMENT")
        return " ".join(parts)

    def create_model(self, model: type[t.Any]) -> None:
        needs_pad = _needs_pad_vector(model)
        columns = [
            self._column_definition(model, field)
            for field in model._meta.local_fields
        ]
        if needs_pad:
            dim = len(PAD_VECTOR_VALUE)
            columns.append(
                f"{self.quote_name(PAD_VECTOR_FIELD)} VECTOR({dim})"
            )
        sql = self.sql_create_table % {
            "table": self.quote_name(model._meta.db_table),
            "definition": ", ".join(columns),
        }
        self.execute(sql)
        if needs_pad and not self.collect_sql:
            # Milvus requires an index *and* a `LOAD` before a
            # collection is queryable at all -- not just before vector
            # search, confirmed directly: `django_migrations` (no real
            # vector field, so the only column to index is the hidden
            # pad column above) raised "collection not loaded" on a
            # plain `SELECT` the moment `MigrationRecorder` queried it,
            # with no vector search involved. A model *with* a real
            # vector field is deliberately left unindexed here (see the
            # module docstring: index/metric is a query-shape decision
            # this generic hook shouldn't guess) -- but there's no
            # equivalent user-code moment to call
            # `create_index_and_load` for Django's own internal
            # bookkeeping models, so this is the one case that has to
            # index itself, on the one column it knows nothing else
            # will ever configure. Skipped under `collect_sql=True`
            # (``sqlmigrate``/dry runs) the same way `self.execute`
            # itself is -- `create_index_and_load` has no SQL-collecting
            # mode of its own and would otherwise execute for real
            # against a run that promised not to.
            create_index_and_load(
                self.connection, model._meta.db_table, PAD_VECTOR_FIELD
            )

    def delete_model(self, model: type[t.Any]) -> None:
        self.execute(
            self.sql_delete_table
            % {"table": self.quote_name(model._meta.db_table)}
        )

    def add_field(self, model: type[t.Any], field: Field) -> None:
        table = self.quote_name(model._meta.db_table)
        column = self._column_definition(model, field)
        self.execute(f"ALTER TABLE {table} ADD FIELD {column}")

    def remove_field(self, model: type[t.Any], field: Field) -> None:
        msg = (
            "Milvus cannot drop a field from an existing collection -- "
            "the collection needs recreating. Same restriction "
            "sqlglot-milvus enforces at the SQL level; see its "
            "ALTER TABLE rejection."
        )
        raise NotImplementedError(msg)

    def alter_field(
        self,
        model: type[t.Any],
        old_field: Field,
        new_field: Field,
        strict: bool = False,
    ) -> None:
        msg = (
            "Milvus cannot change a field's type or a vector's "
            "dimension in place -- the collection needs recreating."
        )
        raise NotImplementedError(msg)


def _with_value(value: t.Any) -> str:
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def create_index_and_load(
    connection: t.Any,
    table: str,
    field_name: str,
    *,
    using: str = "HNSW",
    metric_type: str = "COSINE",
    **index_params: t.Any,
) -> None:
    """The explicit follow-up step ``create_model`` deliberately
    doesn't do automatically (see the module docstring). Call once,
    after defining the model, before querying it."""
    options = {"metric_type": metric_type, **index_params}
    knobs = ", ".join(f"{k}={_with_value(v)}" for k, v in options.items())
    with connection.cursor() as cursor:
        cursor.execute(
            f'CREATE INDEX idx_{field_name} ON "{table}" ("{field_name}") '
            f"USING {using} WITH ({knobs})"
        )
        cursor.execute(f'LOAD TABLE "{table}"')


__all__ = ["DatabaseSchemaEditor", "create_index_and_load"]
