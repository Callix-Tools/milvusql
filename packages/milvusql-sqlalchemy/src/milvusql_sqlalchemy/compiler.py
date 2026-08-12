"""Compilers for the ``milvusql`` dialect.

Deliberately minimal (same lesson as ``elasticsearch-dbapi``'s
``BaseESCompiler``, read directly during design: it overrides almost
nothing because ANSI SQL and its target language overlap heavily).
MilvusQL is SQL-shaped by construction (Track A's own README: "on the
SQL surface that is the ordinary SQL word, so CREATE TABLE/DROP TABLE
parse for free"), and verified directly: base ``SQLCompiler``'s
``LIMIT n OFFSET m`` and base ``DDLCompiler``'s
``CREATE INDEX name ON table (cols)`` already match Track A's grammar
byte for byte. Only what's genuinely different gets overridden: the
``WITH (...)``/``USING <method>`` suffixes MilvusQL adds to
``CREATE TABLE``/``CREATE INDEX``, and rendering ``hybrid.py``'s
``HYBRID SEARCH`` construct (D3) -- nothing else needs touching.
"""

from __future__ import annotations

import typing as t

from sqlalchemy.sql import compiler

from milvusql_sqlalchemy.hybrid import HybridSearchClause

if t.TYPE_CHECKING:
    from sqlalchemy.sql.schema import Table

    from milvusql_sqlalchemy.hybrid import SearchArm


def _sql_string_literal(value: str) -> str:
    """MilvusQL string literals are single-quoted (verified directly:
    a double-quoted value in a ``WITH (...)`` property parses as a
    quoted *identifier*, ``exp.Var``, not ``exp.Literal`` -- Track A's
    grammar is ANSI SQL here, where ``"..."`` means identifier and
    ``'...'`` means string). Embedded quotes double up, the standard
    SQL escape, same as everywhere else in this dialect's text."""
    return "'" + value.replace("'", "''") + "'"


def _render_with(options: dict[str, t.Any]) -> str:
    pairs = []
    for key, value in options.items():
        if value is None:
            continue
        rendered = (
            _sql_string_literal(value)
            if isinstance(value, str)
            else str(value)
        )
        pairs.append(f"{key}={rendered}")
    return f" WITH ({', '.join(pairs)})" if pairs else ""


class MilvusDDLCompiler(compiler.DDLCompiler):
    def get_column_specification(self, column: t.Any, **kwargs: t.Any) -> str:
        """Milvus has exactly one primary key column, always inline
        (Track A's own examples: ``id BIGINT PRIMARY KEY
        AUTO_INCREMENT``), never a separate ``PRIMARY KEY (...)``
        constraint -- unlike the base compiler, which by default emits
        the latter (confirmed directly: plain ``primary_key=True`` on
        an ``Integer`` column produces a trailing ``PRIMARY KEY (id)``
        line, not an inline keyword)."""
        colspec = super().get_column_specification(column, **kwargs)
        if column.primary_key:
            colspec += " PRIMARY KEY"
            if (
                column.table is not None
                and column is column.table._autoincrement_column
            ):
                colspec += " AUTO_INCREMENT"
        return colspec

    def post_create_table(self, table: Table) -> str:
        options = table.dialect_options["milvusql"]
        return _render_with(
            {
                "shards": options.get("shards"),
                "consistency_level": options.get("consistency_level"),
                "partition_key": options.get("partition_key"),
            }
        )

    def visit_create_index(
        self,
        create: t.Any,
        include_schema: bool = False,
        include_table_schema: bool = True,
        **kw: object,
    ) -> str:
        text = super().visit_create_index(
            create,
            include_schema=include_schema,
            include_table_schema=include_table_schema,
            **kw,
        )
        index = create.element
        options = index.dialect_options["milvusql"]
        using = options.get("using")
        if using:
            text += f" USING {using}"
        text += _render_with(options.get("with") or {})
        return text


class MilvusSQLCompiler(compiler.SQLCompiler):
    """Base ``SELECT``/``INSERT``/``DELETE`` rendering already matches
    MilvusQL untouched; the two overrides below exist only for
    ``hybrid.py``'s ``HYBRID SEARCH`` construct (D3), which has no
    equivalent in any base SQL dialect this compiler could inherit."""

    def visit_milvus_search_arm(self, arm: SearchArm, **kw: t.Any) -> str:
        return f"{self.process(arm.distance, **kw)} WEIGHT {arm.weight}"

    def visit_milvus_hybrid_search(
        self, clause: HybridSearchClause, **kw: t.Any
    ) -> str:
        arms_sql = ", ".join(self.process(arm, **kw) for arm in clause.arms)
        params_sql = ", ".join(
            f"{k}={v}" for k, v in clause.rerank_params.items()
        )
        rerank_sql = (
            f"{clause.rerank}({params_sql})" if params_sql else clause.rerank
        )
        return f"HYBRID SEARCH ({arms_sql}) RERANK {rerank_sql}"

    def order_by_clause(self, select: t.Any, **kw: t.Any) -> str:
        """MilvusQL has no ``ORDER BY`` keyword for hybrid search --
        ``HYBRID SEARCH ... RERANK ...`` sits in exactly the position
        ``ORDER BY`` would (D3's own reasoning in ``hybrid.py``), so a
        lone ``HybridSearchClause`` passed to ``.order_by()`` is
        rendered there directly, without the keyword."""
        order_by = select._order_by_clauses
        if len(order_by) == 1 and isinstance(order_by[0], HybridSearchClause):
            return " " + self.process(order_by[0], **kw)
        return super().order_by_clause(select, **kw)


__all__ = ["MilvusDDLCompiler", "MilvusSQLCompiler"]
