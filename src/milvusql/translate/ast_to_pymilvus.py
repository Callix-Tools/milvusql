"""Dispatch table: parsed MilvusQL AST -> a ``pymilvus`` call to make.

Deliberately call-style-agnostic (D1/D12): every ``_build_*`` function
here is a plain, synchronous function that returns a :class:`Call`
describing *what* to call, never calling the network-touching method
itself. ``dbapi/cursor.py`` (sync) and ``aio.py`` (async) are the only
two places that actually invoke ``getattr(client, call.method)`` --
the sync cursor calls it directly, the async client ``await``-s it.
That split is why this module takes no position on sync vs. async: a
``MilvusClient``/``AsyncMilvusClient`` pair share every method name and
argument shape (verified against pymilvus 2.6), so one dispatch table
serves both.

``client`` is still a required argument to :func:`build_call`, not
because this module performs I/O, but because ``create_schema()`` and
``prepare_index_params()`` are the only supported way to build the
objects ``create_collection``/``create_index`` expect, and both are
plain, non-blocking helpers on the client (confirmed on
``AsyncMilvusClient`` too: ``inspect.iscoroutinefunction`` is ``False``
for both) -- calling them here does not compromise the sync/async split.
"""

from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass, field

from pymilvus import AnnSearchRequest, DataType, RRFRanker, WeightedRanker
from sqlglot import exp
from sqlglot_milvus.expressions import (
    CONSISTENCY_ARG,
    HYBRID_ARG,
    METRIC_TYPES,
    SEARCH_PARAMS_ARG,
    AddField,
    HybridSearch,
    LoadTable,
    ReleaseTable,
)

from milvusql.dbapi import errors

if t.TYPE_CHECKING:
    from pymilvus import AsyncMilvusClient, MilvusClient

#: Milvus's own hard ceiling on how many rows a single ``query``/
#: ``search`` RPC can return (confirmed directly: ``pymilvus.orm.
#: constants.MAX_BATCH_SIZE``). Used as the *default* ``limit`` for a
#: SELECT that carries no ``LIMIT`` clause of its own -- Django's own
#: SQL compiler never emits a ``LIMIT`` for a bare ``Model.objects.
#: all()`` (confirmed against ``DatabaseOperations.no_limit_value()``
#: returning ``None``, which suppresses the clause entirely), so a
#: small hand-picked default here used to silently drop every row past
#: it with no error at all. This is Milvus's actual per-call ceiling,
#: not a guess -- the honest answer to "how many rows can come back
#: with no explicit LIMIT" is "as many as Milvus itself allows in one
#: call", never an arbitrary smaller number.
_DEFAULT_QUERY_LIMIT = 16384

#: Rows as PEP 249 wants them back from ``fetch*``, the column
#: descriptions ``Cursor.description`` exposes (name-only; the rest of
#: the 7-tuple is unknown and left ``None``, same simplification
#: ``elasticsearch-dbapi`` makes for the same reason -- Milvus's client
#: gives us values, not a typed wire schema, at this layer), the
#: ``rowcount`` PEP 249 wants set after ``execute()`` (-1 for DDL/DDL-like
#: statements where the concept does not apply, same as every other
#: DBAPI does for ``CREATE``/``LOAD``), and ``lastrowid`` -- the
#: PEP 249-conventional (sqlite3, MySQLdb, ...) place an auto-assigned
#: primary key surfaces after ``INSERT``, ``None`` for anything else.
RowsAndDescription = tuple[
    list[tuple[t.Any, ...]], list[tuple] | None, int, t.Any
]

Postprocess = t.Callable[[t.Any], RowsAndDescription]


@dataclass(frozen=True)
class Call:
    """What to call on a ``MilvusClient``/``AsyncMilvusClient``, and how
    to turn its return value into DBAPI-shaped rows.

    ``then`` exists for the one statement that genuinely cannot be a
    single RPC: ``UPDATE``. Milvus has no partial-row update, only
    ``upsert()``, which replaces the whole entity -- so an ``UPDATE ...
    SET ... WHERE ...`` has to read the matching rows in full first,
    merge the ``SET`` values in Python, then upsert the merged rows
    back. Given this call's raw result, ``then`` returns the next
    ``Call`` to run, or ``None`` to stop and postprocess *this* call's
    raw result instead -- still just describing what to call next, the
    same "no I/O in this module" invariant every ``_build_*`` function
    above keeps; ``dbapi/cursor.py`` (sync) and ``aio.py`` (async) are
    still the only two places that actually invoke a client method,
    now in a small loop instead of once."""

    method: str
    kwargs: dict[str, t.Any] = field(default_factory=dict)
    postprocess: Postprocess = lambda _raw: ([], None, -1, None)
    then: t.Callable[[t.Any], Call | None] | None = None


def _no_rows(_raw: t.Any) -> RowsAndDescription:  # noqa: ANN401
    return [], None, -1, None


def _insert_result(raw: dict[str, t.Any]) -> RowsAndDescription:
    """``insert`` returns ``{"insert_count": n, "ids": [...]}`` --
    confirmed directly against Milvus Lite, ``ids`` holds the
    server-assigned primary keys for an ``auto_id`` collection. DBAPI
    convention (sqlite3, MySQLdb) is ``cursor.lastrowid`` -- the most
    recently inserted row's key -- which is the last element for a
    multi-row ``execute()``, matching those DBAPIs' own semantics for a
    single ``executemany``-shaped call."""
    ids = raw.get("ids")
    return [], None, raw.get("insert_count", -1), ids[-1] if ids else None


def _mutation_count(key: str) -> Postprocess:
    """``delete`` usually returns ``{"delete_count": n}`` but --
    confirmed directly against Milvus Lite -- falls back to a bare list
    of deleted primary keys on servers old enough to still return them
    (``MilvusClient.delete``'s own compatibility branch); DBAPI wants a
    single ``rowcount`` either way, not a fetchable row."""

    def postprocess(raw: dict[str, t.Any] | list[t.Any]) -> RowsAndDescription:
        if isinstance(raw, list):
            return [], None, len(raw), None
        return [], None, raw.get(key, -1), None

    return postprocess


def _update_zero(_raw: t.Any) -> RowsAndDescription:  # noqa: ANN401
    """``UPDATE``'s first step (the read) short-circuits here when
    nothing matched the ``WHERE`` clause -- ``rowcount`` is honestly
    ``0``, not the ``-1`` "not applicable" DDL convention, since an
    UPDATE that matched nothing is a completed, well-understood
    outcome."""
    return [], None, 0, None


def _upsert_result(raw: dict[str, t.Any]) -> RowsAndDescription:
    """``UPDATE``'s second step (the write): ``upsert`` returns
    ``{"upsert_count": n, ...}`` -- confirmed directly against Milvus
    Lite, the same shape family as ``insert``'s ``insert_count``."""
    return [], None, raw.get("upsert_count", -1), None


# -----------------------------------------------------------------------
# Shared value/property helpers
# -----------------------------------------------------------------------


def _resolve_value(
    node: exp.Expression, parameters: dict[str, t.Any]
) -> t.Any:  # noqa: ANN401 -- a bind value is genuinely any JSON-ish type
    """A literal, placeholder or array -> the Python value it names.
    Bind values (vectors, scalars passed as ``:name``) are read from
    ``parameters``, never parsed back out of query text (D1)."""
    if isinstance(node, (exp.Placeholder, exp.Parameter)):
        name = node.this
        if name not in parameters:
            msg = f"missing bind parameter {name!r}"
            raise errors.ProgrammingError(msg)
        return parameters[name]
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.Literal):
        if node.is_string:
            return node.this
        text = node.this
        return float(text) if "." in text or "e" in text.lower() else int(text)
    if isinstance(node, (exp.Array, exp.Tuple)):
        return [_resolve_value(e, parameters) for e in node.expressions]
    if isinstance(node, exp.Null):
        return None
    msg = f"unsupported value expression: {node.__class__.__name__}"
    raise errors.NotSupportedError(msg)


def _property_list_to_dict(
    properties: list[exp.Property], parameters: dict[str, t.Any]
) -> dict[str, t.Any]:
    return {
        prop.this.name: _resolve_value(prop.args["value"], parameters)
        for prop in properties
    }


def _properties_to_dict(
    node: exp.Properties | None, parameters: dict[str, t.Any]
) -> dict[str, t.Any]:
    if node is None:
        return {}
    return _property_list_to_dict(node.expressions, parameters)


# -----------------------------------------------------------------------
# WHERE -> Milvus filter-expression templating (shared: DELETE, SELECT)
# -----------------------------------------------------------------------

_COMPARISON_OPS: dict[type[exp.Expression], str] = {
    exp.EQ: "==",
    exp.NEQ: "!=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
}


def _render_filter_value(value: t.Any) -> str:  # noqa: ANN401
    """A resolved bind value -> its Milvus filter-DSL literal spelling.
    ``json.dumps`` for strings gives correct escaping (quotes,
    backslashes, control characters) for free -- Milvus's own string
    literal syntax is JSON-compatible double-quoting, confirmed against
    the docs and against a real Milvus Lite instance below."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    msg = f"unsupported filter value type: {type(value).__name__}"
    raise errors.NotSupportedError(msg)


def _render_filter(  # noqa: PLR0911, PLR0912 -- one return per AST node case, clearer flat than nested
    node: exp.Expression, parameters: dict[str, t.Any]
) -> str:
    """MilvusQL's WHERE tree -> Milvus's boolean filter-expression text.

    Milvus's ``{name}`` filter templating (the documented way to keep
    bound values out of the expression string) was tried first and
    empirically does not work against Milvus Lite -- confirmed with a
    direct ``MilvusClient.query(filter="x == {p}", filter_params=...)``
    call, which the embedded server's expression parser rejects with
    "unexpected character '{'". Scalar WHERE values (unlike vectors,
    the thing D1's never-inline-into-text rule is actually about) are
    small and not precision-sensitive, so resolving and escaping them
    inline here, via :func:`_render_filter_value`, is the fallback that
    is verified to actually work."""
    if isinstance(node, exp.Where):
        return _render_filter(node.this, parameters)
    if isinstance(node, exp.Paren):
        return f"({_render_filter(node.this, parameters)})"
    if isinstance(node, exp.And):
        left = _render_filter(node.this, parameters)
        right = _render_filter(node.expression, parameters)
        return f"({left} and {right})"
    if isinstance(node, exp.Or):
        left = _render_filter(node.this, parameters)
        right = _render_filter(node.expression, parameters)
        return f"({left} or {right})"
    if isinstance(node, exp.Not):
        return f"not ({_render_filter(node.this, parameters)})"
    op = _COMPARISON_OPS.get(type(node))
    if op is not None:
        left = _render_filter(node.this, parameters)
        right = _render_filter(node.expression, parameters)
        return f"{left} {op} {right}"
    if isinstance(node, exp.In):
        # Milvus's filter DSL has its own native `field in [...]`
        # syntax (confirmed directly against Milvus Lite) -- no
        # transpiling needed, just render the column and each
        # resolved value the same way every other comparison here
        # does. `col IN ()` (Django's `filter(pk__in=[])`, though the
        # ORM usually short-circuits that to an empty queryset before
        # any SQL is even sent) matches nothing, same as SQL's own
        # empty-IN-list semantics -- `false` says that honestly rather
        # than sending Milvus a syntax error over an empty `[]`.
        if not node.expressions:
            return "false"
        column = _render_filter(node.this, parameters)
        values = ", ".join(
            _render_filter_value(_resolve_value(value, parameters))
            for value in node.expressions
        )
        return f"{column} in [{values}]"
    if isinstance(node, exp.Like):
        # Milvus's filter DSL has its own native `field like "pattern"`
        # syntax (confirmed directly against Milvus Lite, SQL-style `%`/
        # `_` wildcards) -- no transpiling needed. `NOT LIKE` compiles
        # to a `negate` flag on the same node (unlike `NOT IN`/
        # `IS NOT NULL`, which wrap in a separate `exp.Not`), so it is
        # handled here rather than falling through to the generic `Not`
        # case above.
        column = _render_filter(node.this, parameters)
        pattern = _render_filter_value(
            _resolve_value(node.expression, parameters)
        )
        text = f"{column} like {pattern}"
        return f"not ({text})" if node.args.get("negate") else text
    if isinstance(node, exp.Between):
        # Milvus's filter DSL has no `BETWEEN` keyword at all (confirmed
        # directly: a bare `id BETWEEN 1 AND 2` is a syntax error), so
        # this transpiles to the equivalent `>=`/`<=` pair every other
        # comparison here already renders -- `NOT BETWEEN` gets this for
        # free via the generic `Not` case above, since (unlike `LIKE`)
        # it wraps in a separate `exp.Not` rather than a node flag.
        column = _render_filter(node.this, parameters)
        low = _render_filter_value(
            _resolve_value(node.args["low"], parameters)
        )
        high = _render_filter_value(
            _resolve_value(node.args["high"], parameters)
        )
        return f"({column} >= {low} and {column} <= {high})"
    if isinstance(node, exp.Is):
        # Milvus's filter DSL only has `is null`/`is not null`
        # (confirmed directly) -- no other `IS <predicate>` form (e.g.
        # `IS TRUE`) to transpile to, so anything but a `NULL` right-hand
        # side is a real gap, not a silent guess.
        if not isinstance(node.expression, exp.Null):
            msg = (
                "unsupported IS comparison: IS "
                f"{node.expression.__class__.__name__}"
            )
            raise errors.NotSupportedError(msg)
        column = _render_filter(node.this, parameters)
        return f"{column} is null"
    if isinstance(node, exp.Column):
        return node.this.name
    if isinstance(
        node, (exp.Placeholder, exp.Parameter, exp.Literal, exp.Boolean)
    ):
        return _render_filter_value(_resolve_value(node, parameters))
    msg = f"unsupported filter expression: {node.__class__.__name__}"
    raise errors.NotSupportedError(msg)


def _filter_text(where: exp.Where | None, parameters: dict[str, t.Any]) -> str:
    if where is None:
        return ""
    return _render_filter(where, parameters)


# -----------------------------------------------------------------------
# DDL: CREATE TABLE / CREATE INDEX
# -----------------------------------------------------------------------

_SCALAR_TYPES: dict[exp.DataType.Type, DataType] = {
    exp.DataType.Type.TINYINT: DataType.INT8,
    exp.DataType.Type.SMALLINT: DataType.INT16,
    exp.DataType.Type.INT: DataType.INT32,
    exp.DataType.Type.MEDIUMINT: DataType.INT32,
    exp.DataType.Type.BIGINT: DataType.INT64,
    exp.DataType.Type.FLOAT: DataType.FLOAT,
    exp.DataType.Type.DOUBLE: DataType.DOUBLE,
    exp.DataType.Type.BOOLEAN: DataType.BOOL,
    exp.DataType.Type.JSON: DataType.JSON,
}


def _map_datatype(dtype: exp.DataType) -> tuple[DataType, dict[str, t.Any]]:
    """A parsed ``ColumnDef.kind`` -> ``(pymilvus DataType, extra
    add_field kwargs)``. Only the types the phase-1 DDL surface actually
    needs (README's ``CREATE TABLE`` example set) are mapped; anything
    else -- ``TEXT``, ``ARRAY``, a binary-vector spelling -- is a real
    gap, not a silent guess, so it raises rather than picks an arbitrary
    Milvus type for a spelling this layer has never seen used.
    """
    if dtype.this is exp.DataType.Type.VARCHAR:
        param = dtype.expressions[0] if dtype.expressions else None
        max_length = int(param.this.this) if param is not None else 65535
        return DataType.VARCHAR, {"max_length": max_length}
    if dtype.this is exp.DataType.Type.VECTOR:
        dim = int(dtype.expressions[0].this.this)
        return DataType.FLOAT_VECTOR, {"dim": dim}
    if dtype.this is exp.DataType.Type.USERDEFINED:
        # ``SPARSEVEC`` has no dedicated sqlglot ``DataType.Type`` of
        # its own (unlike ``VECTOR``, recycled from a real one) -- it
        # parses as a generic user-defined type carrying its own
        # spelling as free text in ``kind`` (confirmed directly against
        # sqlglot's own parser output), case-preserved from whatever the
        # caller wrote, hence the case-insensitive match.
        kind = dtype.args.get("kind")
        if isinstance(kind, str) and kind.upper() == "SPARSEVEC":
            return DataType.SPARSE_FLOAT_VECTOR, {}
    mapped = _SCALAR_TYPES.get(dtype.this)
    if mapped is None:
        msg = f"unsupported column type: {dtype.sql(dialect='milvus')}"
        raise errors.NotSupportedError(msg)
    return mapped, {}


def _build_create_table(
    client: MilvusClient | AsyncMilvusClient,
    ast: exp.Create,
    parameters: dict[str, t.Any],
) -> Call:
    schema_node = ast.this
    table_name = schema_node.this.name
    props = _properties_to_dict(ast.args.get("properties"), parameters)

    milvus_schema = client.create_schema(
        enable_dynamic_field=False,
        partition_key_field=props.pop("partition_key", None),
    )
    columns = [
        c for c in schema_node.expressions if isinstance(c, exp.ColumnDef)
    ]
    # A standalone `PRIMARY KEY (col)` table constraint -- standard SQL,
    # and what SQLAlchemy's own DDLCompiler emits by default instead of
    # an inline `PRIMARY KEY` on the column -- names the primary column
    # separately from its ColumnDef; verified directly against
    # `metadata.create_all()`'s actual output.
    pk_names = {
        ident.name
        for node in schema_node.expressions
        if isinstance(node, exp.PrimaryKey)
        for ident in node.expressions
    }
    for column in columns:
        name = column.this.name
        milvus_type, extra = _map_datatype(column.kind)
        constraint_kinds = {type(c.kind) for c in (column.constraints or [])}
        is_primary = (
            exp.PrimaryKeyColumnConstraint in constraint_kinds
            or name in pk_names
        )
        # A `Mapped[T | None]` column (no explicit `NOT NULL`, what
        # SQLAlchemy's own DDL compiler emits for it) needs
        # `nullable=True` on the Milvus `FieldSchema`, or `pymilvus`
        # rejects a Python `None` at insert time with "FieldData 'x' has
        # 0 rows, expected 1" -- confirmed directly. Primary keys and
        # vector fields can't be nullable in Milvus, so this only
        # applies to ordinary scalar columns.
        nullable = (
            not is_primary
            and exp.NotNullColumnConstraint not in constraint_kinds
            and milvus_type
            not in (DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR)
        )
        milvus_schema.add_field(
            field_name=name,
            datatype=milvus_type,
            is_primary=is_primary,
            auto_id=exp.AutoIncrementColumnConstraint in constraint_kinds,
            nullable=nullable,
            **extra,
        )

    kwargs: dict[str, t.Any] = {
        "collection_name": table_name,
        "schema": milvus_schema,
    }
    if "shards" in props:
        kwargs["num_shards"] = props.pop("shards")
    kwargs.update(props)  # consistency_level and anything else pass through
    return Call("create_collection", kwargs, _no_rows)


def _build_create_index(
    client: MilvusClient | AsyncMilvusClient,
    ast: exp.Create,
    parameters: dict[str, t.Any],
) -> Call:
    """Known gap, not in this function: ``AsyncMilvusClient.create_index``
    waits for completion via an ``AllocTimestamp`` RPC that Milvus
    Lite's *async* gRPC server does not implement (confirmed directly;
    the sync server handles the equivalent sync call fine). Nothing to
    fix here -- this dispatch code is identical for both clients, only
    ``await`` differs at the call site -- it is a Milvus Lite server
    limitation for this one RPC, tracked in the test suite rather than
    routed around here."""
    index_node = ast.this
    table_name = index_node.args["table"].name
    index_name = index_node.this.name
    params_node = index_node.args["params"]
    using = params_node.args["using"].name
    columns = [c.this.this.name for c in params_node.args["columns"]]
    knobs = _property_list_to_dict(
        params_node.args.get("with_storage") or [], parameters
    )

    index_params = client.prepare_index_params()
    for field_name in columns:
        index_params.add_index(
            field_name=field_name,
            index_type=using,
            index_name=index_name,
            **knobs,
        )
    return Call(
        "create_index",
        {"collection_name": table_name, "index_params": index_params},
        _no_rows,
    )


def _build_load_table(ast: LoadTable, parameters: dict[str, t.Any]) -> Call:
    props = _properties_to_dict(ast.args.get("properties"), parameters)
    kwargs: dict[str, t.Any] = {"collection_name": ast.this.name}
    if "replicas" in props:
        kwargs["replica_number"] = props["replicas"]
    return Call("load_collection", kwargs, _no_rows)


def _build_release_table(ast: ReleaseTable) -> Call:
    return Call(
        "release_collection", {"collection_name": ast.this.name}, _no_rows
    )


def _build_drop_table(ast: exp.Drop) -> Call:
    return Call(
        "drop_collection", {"collection_name": ast.this.name}, _no_rows
    )


def _build_alter_add_field(ast: exp.Alter) -> Call:
    """``ALTER TABLE items ADD FIELD tag VARCHAR(32)``.

    sqlglot-milvus's own grammar rejects every other ``ALTER`` action
    (``DROP``/``ALTER COLUMN``/``MODIFY``) at *parse* time with a
    ``ParseError`` (Milvus can't perform them at all), so the single
    action reaching here is always an ``AddField`` -- the ``isinstance``
    check below is defense in depth against a synthetic or
    foreign-dialect AST, not a real branch the grammar can produce.

    ``MilvusClient.add_collection_field`` returns ``UNIMPLEMENTED``
    against Milvus Lite (verified directly -- a raw ``grpc.RpcError``,
    not even a ``MilvusException``), which ``errors.translate`` already
    maps to :class:`~milvusql.dbapi.errors.NotSupportedError` (see
    ``_translate_grpc_error``). So on Lite this still raises the same
    exception type it always has -- now for real, from the RPC layer,
    rather than a blanket pre-emptive rejection that would also have
    blocked a real Milvus server that *does* implement it.

    A newly added field always needs ``nullable=True``: existing rows
    have no value to backfill for it, and ``pymilvus`` itself requires
    it for a vector field (raises ``ParamError`` otherwise) -- setting
    it unconditionally is correct for every column kind, not just
    vectors.
    """
    actions = ast.args.get("actions") or []
    if len(actions) != 1 or not isinstance(actions[0], AddField):
        msg = "unsupported ALTER TABLE action"
        raise errors.NotSupportedError(msg)
    column = actions[0].this
    milvus_type, extra = _map_datatype(column.kind)
    kwargs: dict[str, t.Any] = {
        "collection_name": ast.this.name,
        "field_name": column.this.name,
        "data_type": milvus_type,
        "nullable": True,
        **extra,
    }
    return Call("add_collection_field", kwargs, _no_rows)


# -----------------------------------------------------------------------
# DML: INSERT / DELETE
# -----------------------------------------------------------------------


def _insert_rows(
    ast: exp.Insert, parameters: dict[str, t.Any]
) -> list[dict[str, t.Any]]:
    """The row dicts one bind-parameter set's ``INSERT`` names, shared
    between a single-row ``execute()`` and the batched
    ``executemany()`` path below -- both need exactly the same
    ``column -> resolved value`` mapping, just assembled from a
    different number of parameter sets."""
    schema_node = ast.this
    columns = [ident.name for ident in schema_node.expressions]
    values_node = ast.args["expression"]
    return [
        {
            name: _resolve_value(value, parameters)
            for name, value in zip(columns, row.expressions, strict=True)
        }
        for row in values_node.expressions
    ]


def _build_insert(ast: exp.Insert, parameters: dict[str, t.Any]) -> Call:
    table_name = ast.this.this.name
    rows = _insert_rows(ast, parameters)
    return Call(
        "insert",
        {"collection_name": table_name, "data": rows},
        _insert_result,
    )


def _build_batch_insert(
    ast: exp.Insert, seq_of_parameters: t.Sequence[dict[str, t.Any]]
) -> Call:
    """``executemany()`` over an ``INSERT`` -- unlike every other
    statement shape, Milvus's own ``insert()`` already accepts a full
    list of rows in one RPC (confirmed directly: ``data=[...]`` is not
    limited to one row), so every bind-parameter set's row(s) are
    merged into a single call instead of one RPC per parameter set.
    ``_insert_result`` needs no changes for this: it already reduces a
    multi-row ``insert()`` response to one ``rowcount``/``lastrowid``
    pair the same way, whether the multiple rows came from one
    multi-VALUES statement or many parameter sets."""
    table_name = ast.this.this.name
    rows = [
        row
        for parameters in seq_of_parameters
        for row in _insert_rows(ast, parameters)
    ]
    return Call(
        "insert",
        {"collection_name": table_name, "data": rows},
        _insert_result,
    )


def build_batch_call(
    ast: exp.Expression, seq_of_parameters: t.Sequence[dict[str, t.Any]]
) -> Call | None:
    """``executemany()``'s entry point: a parsed statement and every
    one of its bind-parameter sets -> a single batched ``Call``, or
    ``None`` when Milvus has no batched primitive for this statement
    shape (``UPDATE``'s ``query``-then-``upsert`` pair and ``DELETE``'s
    single-filter shape don't merge across independent parameter sets
    the way ``INSERT``'s row list does) -- the caller then falls back
    to one :func:`build_call` per parameter set, same as before this
    existed."""
    if isinstance(ast, exp.Insert):
        return _build_batch_insert(ast, seq_of_parameters)
    return None


def _build_delete(ast: exp.Delete, parameters: dict[str, t.Any]) -> Call:
    table_name = ast.this.name
    filter_text = _filter_text(ast.args.get("where"), parameters)
    kwargs: dict[str, t.Any] = {"collection_name": table_name}
    if filter_text:
        kwargs["filter"] = filter_text
    return Call("delete", kwargs, _mutation_count("delete_count"))


def _build_update(ast: exp.Update, parameters: dict[str, t.Any]) -> Call:
    """``UPDATE table SET col = val[, ...] WHERE ...`` -- see
    :class:`Call`'s docstring for why this is a read (``query``) whose
    ``then`` chains into a write (``upsert``), not one RPC."""
    table_name = ast.this.name
    # `UPDATE`'s `SET` list is always `col = <expr>` (`exp.EQ`) by
    # grammar -- `_resolve_value` is what actually rejects an
    # unsupported right-hand side (`SET stock = stock + 1`, an `F()`
    # expression in Django terms: `exp.Add`, not a bare bind value or
    # literal), the same way it rejects one anywhere else in this
    # module.
    set_values = {
        assignment.this.name: _resolve_value(assignment.expression, parameters)
        for assignment in ast.expressions
    }
    filter_text = _filter_text(ast.args.get("where"), parameters)

    query_kwargs: dict[str, t.Any] = {
        "collection_name": table_name,
        # Every field, vectors included -- `upsert()` replaces the
        # whole entity, so the merged row has to carry everything the
        # row already had, not just the columns SET touches.
        "output_fields": ["*"],
        "limit": _DEFAULT_QUERY_LIMIT,
    }
    if filter_text:
        query_kwargs["filter"] = filter_text

    def _then(raw: list[dict[str, t.Any]]) -> Call | None:
        if not raw:
            return None
        merged = [{**row, **set_values} for row in raw]
        return Call(
            "upsert",
            {"collection_name": table_name, "data": merged},
            _upsert_result,
        )

    return Call("query", query_kwargs, _update_zero, then=_then)


# -----------------------------------------------------------------------
# DQL: SELECT (vector search and plain filter query)
# -----------------------------------------------------------------------


def _select_field_names(ast: exp.Select) -> list[str]:
    """The real Milvus field names to request via ``output_fields`` --
    unwraps ``t.col AS alias`` down to ``col``. Deliberately distinct
    from :func:`_select_output_names`: SQLAlchemy's ORM always labels
    every column (``SELECT t.id AS t_id, ...``, confirmed directly in
    ``Session.get()``'s emitted SQL), and asking Milvus for a field
    literally named ``t_id`` returns nothing for it -- silently
    (Milvus's ``query``/``search`` just omit an unknown output field,
    they don't error), which used to make every ORM load return rows
    of ``None``s."""
    return [
        column.this.name if isinstance(column, exp.Alias) else column.name
        for column in ast.expressions
    ]


def _select_output_names(ast: exp.Select) -> list[str]:
    """The labels to report back through ``cursor.description`` and to
    build result tuples with -- the alias when there is one, the bare
    column name otherwise."""
    return [column.output_name for column in ast.expressions]


def _description(output_names: list[str]) -> list[tuple]:
    return [
        (name, None, None, None, None, None, True) for name in output_names
    ]


def _search_rows(
    field_names: list[str], output_names: list[str]
) -> Postprocess:
    def postprocess(raw: list[list[dict[str, t.Any]]]) -> RowsAndDescription:
        hits = raw[0] if raw else []
        rows = []
        for hit in hits:
            available = {
                **hit.get("entity", {}),
                "id": hit.get("id"),
                "distance": hit.get("distance"),
            }
            rows.append(tuple(available.get(name) for name in field_names))
        return rows, _description(output_names), len(rows), None

    return postprocess


def _query_rows(
    field_names: list[str], output_names: list[str]
) -> Postprocess:
    def postprocess(raw: list[dict[str, t.Any]]) -> RowsAndDescription:
        rows = [tuple(row.get(name) for name in field_names) for row in raw]
        return rows, _description(output_names), len(rows), None

    return postprocess


def _sorted_query_rows(
    field_names: list[str],
    output_names: list[str],
    order_keys: list[tuple[str, bool]],
    limit: int,
) -> Postprocess:
    """Client-side ``ORDER BY <scalar column>`` + ``LIMIT``: Milvus's
    ``query()`` RPC has no ordering concept of its own (confirmed --
    it accepts a filter and a row cap, nothing else), so this sorts
    every matching row (fetched up to :data:`_DEFAULT_QUERY_LIMIT`,
    same ceiling a plain unordered SELECT is already subject to) in
    Python and only *then* applies the real ``LIMIT`` -- sorting after
    truncating would return the wrong rows entirely."""

    def postprocess(raw: list[dict[str, t.Any]]) -> RowsAndDescription:
        rows = list(raw)
        # Stable-sort by the least significant key first so the most
        # significant key (applied last) wins ties -- the standard way
        # to build a multi-key sort out of single-key stable sorts.
        for column, desc in reversed(order_keys):
            rows.sort(
                key=lambda row, column=column: (
                    row.get(column) is None,
                    row.get(column),
                ),
                reverse=desc,
            )
        rows = rows[:limit]
        tuples = [tuple(row.get(name) for name in field_names) for row in rows]
        return tuples, _description(output_names), len(tuples), None

    return postprocess


#: Django's `Sum`/`Avg`/`Min`/`Max`/`Count` all compile to one of
#: these five standard SQL aggregate functions -- Milvus computes none
#: of them server-side (`count(*)` is the one documented exception,
#: handled separately below), so every one of these gets reduced from
#: fetched rows in Python.
_AGG_FUNCS: dict[type[exp.Expression], str] = {
    exp.Count: "count",
    exp.Sum: "sum",
    exp.Avg: "avg",
    exp.Min: "min",
    exp.Max: "max",
}


def _table_name(ast: exp.Select) -> str:
    """The single collection a ``SELECT`` reads from.

    Milvus has no cross-collection JOIN and no subquery-as-source, so
    both are rejected explicitly here rather than silently narrowing to
    whatever ``.this.name`` happens to resolve to: a ``JOIN`` was
    previously dropped without a word (the query just ran against the
    first table, as if the ``JOIN`` had never been written), and a
    subquery in ``FROM`` produced an unclear ``AttributeError`` deep in
    ``pymilvus`` instead of a clean, actionable error at translate
    time."""
    if ast.args.get("joins"):
        msg = "JOIN is not supported: Milvus has no cross-collection join"
        raise errors.NotSupportedError(msg)
    from_this = ast.args["from_"].this
    if not isinstance(from_this, exp.Table):
        msg = (
            "SELECT ... FROM <subquery> is not supported: FROM must "
            "name a single collection"
        )
        raise errors.NotSupportedError(msg)
    return from_this.name


def _unwrap_alias(node: exp.Expression) -> exp.Expression:
    return node.this if isinstance(node, exp.Alias) else node


def _is_aggregate_select(ast: exp.Select) -> bool:
    """A bare ``SELECT <agg>(...), ...`` with no ``GROUP BY`` -- the
    shape Django's ``QuerySet.count()``/``aggregate()`` always compile
    to (confirmed directly: neither ever emits a ``GROUP BY``, since
    both return one row, not one row per group)."""
    if ast.args.get("group") is not None or not ast.expressions:
        return False
    return all(type(_unwrap_alias(e)) in _AGG_FUNCS for e in ast.expressions)


def _count_star_rows(count: int, output_names: list[str]) -> Postprocess:
    def postprocess(raw: list[dict[str, t.Any]]) -> RowsAndDescription:
        total = raw[0]["count(*)"] if raw else 0
        # A pure `COUNT(*)`-only select (Django's `.count()`, or
        # `.aggregate(Count("*"), Count("*"))`) always repeats the same
        # value across every expression -- there is nothing else it
        # could mean with no other aggregate or real column present.
        row = tuple(total for _ in range(count))
        return [row], _description(output_names), 1, None

    return postprocess


def _reduce_aggregate_rows(
    specs: list[tuple[str, str | None]], output_names: list[str]
) -> Postprocess:
    def postprocess(raw: list[dict[str, t.Any]]) -> RowsAndDescription:
        values: list[t.Any] = []
        for func, column in specs:
            if func == "count":
                values.append(
                    len(raw)
                    if column is None
                    else sum(1 for row in raw if row.get(column) is not None)
                )
                continue
            numbers = [
                row[column]
                for row in raw
                if column is not None and row.get(column) is not None
            ]
            if func == "sum":
                values.append(sum(numbers) if numbers else 0)
            elif func == "avg":
                values.append(sum(numbers) / len(numbers) if numbers else None)
            elif func == "min":
                values.append(min(numbers) if numbers else None)
            elif func == "max":
                values.append(max(numbers) if numbers else None)
        return [tuple(values)], _description(output_names), 1, None

    return postprocess


def _build_aggregate(ast: exp.Select, parameters: dict[str, t.Any]) -> Call:
    table_name = _table_name(ast)
    filter_text = _filter_text(ast.args.get("where"), parameters)
    output_names = _select_output_names(ast)

    specs: list[tuple[str, str | None]] = []
    for e in ast.expressions:
        inner = _unwrap_alias(e)
        func = _AGG_FUNCS[type(inner)]
        column = None if isinstance(inner.this, exp.Star) else inner.this.name
        specs.append((func, column))

    if all(func == "count" and column is None for func, column in specs):
        # Milvus computes `count(*)` server-side -- no row fetch
        # needed at all, and no ceiling to worry about either
        # (confirmed directly against Milvus Lite: `query(...,
        # output_fields=["count(*)"])` returns the true total even
        # past `_DEFAULT_QUERY_LIMIT` rows).
        kwargs: dict[str, t.Any] = {
            "collection_name": table_name,
            "output_fields": ["count(*)"],
        }
        if filter_text:
            kwargs["filter"] = filter_text
        return Call(
            "query", kwargs, _count_star_rows(len(specs), output_names)
        )

    columns = sorted({column for _, column in specs if column is not None})
    kwargs = {
        "collection_name": table_name,
        "output_fields": columns or ["count(*)"],
        "limit": _DEFAULT_QUERY_LIMIT,
    }
    if filter_text:
        kwargs["filter"] = filter_text
    return Call("query", kwargs, _reduce_aggregate_rows(specs, output_names))


def _select_order_keys(
    order: exp.Order, field_names: list[str]
) -> list[tuple[str, bool]]:
    """A plain-column ``ORDER BY`` is the common case, but Django's own
    compiler emits an ordinal position (``ORDER BY 1 DESC``) instead of
    repeating the expression whenever the query also carries
    ``.values()``/``.values_list()`` (confirmed directly) -- resolve
    that back to the real field name via the already-computed SELECT
    list rather than rejecting it as unsupported."""
    keys = []
    for ordered in order.expressions:
        node = ordered.this
        desc = bool(ordered.args.get("desc"))
        if isinstance(node, exp.Literal) and node.is_int:
            index = int(node.this) - 1
            if not 0 <= index < len(field_names):
                msg = f"ORDER BY position {node.this} is out of range"
                raise errors.NotSupportedError(msg)
            keys.append((field_names[index], desc))
            continue
        if not isinstance(node, exp.Column):
            msg = f"unsupported ORDER BY expression: {node.__class__.__name__}"
            raise errors.NotSupportedError(msg)
        keys.append((node.name, desc))
    return keys


def _build_hybrid_search(
    hybrid: HybridSearch,
    ast: exp.Select,
    parameters: dict[str, t.Any],
) -> Call:
    table_name = _table_name(ast)
    field_names = _select_field_names(ast)
    output_names = _select_output_names(ast)
    filter_text = _filter_text(ast.args.get("where"), parameters)
    limit_node = ast.args.get("limit")
    limit = (
        int(_resolve_value(limit_node.expression, parameters))
        if limit_node
        else _DEFAULT_QUERY_LIMIT
    )

    reqs = []
    weights = []
    for arm in hybrid.expressions:
        distance_node = arm.this
        metric_type = METRIC_TYPES.get(type(distance_node))
        if metric_type is None:
            msg = (
                "unsupported HYBRID SEARCH arm scoring expression: "
                f"{distance_node.__class__.__name__}"
            )
            raise errors.NotSupportedError(msg)
        column_name = distance_node.this.this.name
        query_vector = _resolve_value(distance_node.expression, parameters)
        weight_node = arm.args.get("weight")
        weight = (
            float(_resolve_value(weight_node, parameters))
            if weight_node is not None
            else 1.0
        )
        weights.append(weight)
        req_kwargs: dict[str, t.Any] = {
            "data": [query_vector],
            "anns_field": column_name,
            "param": {"metric_type": metric_type},
            "limit": limit,
        }
        if filter_text:
            req_kwargs["expr"] = filter_text
        reqs.append(AnnSearchRequest(**req_kwargs))

    rerank_node = hybrid.args.get("rerank")
    kind = rerank_node.this.name.upper() if rerank_node else "RRF"
    rerank_params = (
        _property_list_to_dict(
            rerank_node.args.get("expressions") or [], parameters
        )
        if rerank_node
        else {}
    )
    if kind == "RRF":
        ranker = RRFRanker(k=int(rerank_params.get("k", 60)))
    elif kind == "WEIGHTED":
        ranker = WeightedRanker(*weights)
    else:
        msg = f"unsupported RERANK strategy: {kind}"
        raise errors.NotSupportedError(msg)

    kwargs: dict[str, t.Any] = {
        "collection_name": table_name,
        "reqs": reqs,
        "ranker": ranker,
        "limit": limit,
        "output_fields": field_names,
    }
    return Call(
        "hybrid_search", kwargs, _search_rows(field_names, output_names)
    )


def _build_select(ast: exp.Select, parameters: dict[str, t.Any]) -> Call:
    hybrid = ast.args.get(HYBRID_ARG)
    if hybrid:
        return _build_hybrid_search(hybrid, ast, parameters)

    if _is_aggregate_select(ast):
        return _build_aggregate(ast, parameters)

    if ast.args.get("group") is not None:
        # `_is_aggregate_select` already returns False whenever GROUP BY
        # is present (grouped or not), so every GROUP BY query falls
        # through to here. Milvus's `query()`/`search()` have no
        # server-side grouping and nothing downstream reduces per-group
        # -- silently falling through to the plain-select path below
        # used to run the query ungrouped and return one row per
        # matching entity instead of one row per group.
        msg = "GROUP BY is not supported"
        raise errors.NotSupportedError(msg)

    table_name = _table_name(ast)
    field_names = _select_field_names(ast)
    output_names = _select_output_names(ast)
    filter_text = _filter_text(ast.args.get("where"), parameters)
    limit_node = ast.args.get("limit")
    # LIMIT is a literal in hand-written MilvusQL, but a bound
    # parameter in SQLAlchemy-generated text by default (confirmed
    # directly: `.limit(5)` compiles to `LIMIT :param_1`, not
    # `LIMIT 5`) -- resolve through the same value path as everything
    # else instead of assuming a literal. No LIMIT clause at all (a
    # bare `Model.objects.all()`) falls back to Milvus's own per-call
    # ceiling, never an arbitrary smaller number -- see
    # `_DEFAULT_QUERY_LIMIT`.
    limit = (
        int(_resolve_value(limit_node.expression, parameters))
        if limit_node
        else _DEFAULT_QUERY_LIMIT
    )

    order = ast.args.get("order")
    if order is not None:
        ordered = order.expressions[0]
        distance_node = ordered.this
        metric_type = METRIC_TYPES.get(type(distance_node))
    else:
        metric_type = None

    if order is None or metric_type is None:
        kwargs: dict[str, t.Any] = {
            "collection_name": table_name,
            "output_fields": field_names,
            "limit": limit,
        }
        if filter_text:
            kwargs["filter"] = filter_text
        if order is None:
            return Call(
                "query", kwargs, _query_rows(field_names, output_names)
            )
        # A plain scalar `ORDER BY` -- not a distance operator, so this
        # isn't a vector search. Fetch every matching row (up to
        # Milvus's own ceiling) and sort client-side instead of
        # rejecting it outright.
        order_keys = _select_order_keys(order, field_names)
        fetch_fields = list(field_names)
        for column, _ in order_keys:
            if column not in fetch_fields:
                fetch_fields.append(column)
        kwargs["output_fields"] = fetch_fields
        kwargs["limit"] = _DEFAULT_QUERY_LIMIT
        return Call(
            "query",
            kwargs,
            _sorted_query_rows(field_names, output_names, order_keys, limit),
        )

    column_name = distance_node.this.this.name
    query_vector = _resolve_value(distance_node.expression, parameters)

    search_params_node = ast.args.get(SEARCH_PARAMS_ARG)
    knobs = _property_list_to_dict(
        search_params_node.expressions if search_params_node else [],
        parameters,
    )

    kwargs = {
        "collection_name": table_name,
        "data": [query_vector],
        "anns_field": column_name,
        "limit": limit,
        "output_fields": field_names,
        "search_params": {"metric_type": metric_type, "params": knobs},
    }
    if filter_text:
        kwargs["filter"] = filter_text
    consistency = ast.args.get(CONSISTENCY_ARG)
    if consistency is not None:
        kwargs["consistency_level"] = consistency.this.name

    return Call("search", kwargs, _search_rows(field_names, output_names))


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

_BUILDERS: dict[type[exp.Expression], t.Callable[..., Call]] = {
    LoadTable: lambda _client, ast, params: _build_load_table(ast, params),
    ReleaseTable: lambda _client, ast, _params: _build_release_table(ast),
    exp.Insert: lambda _client, ast, params: _build_insert(ast, params),
    exp.Delete: lambda _client, ast, params: _build_delete(ast, params),
    exp.Update: lambda _client, ast, params: _build_update(ast, params),
    exp.Select: lambda _client, ast, params: _build_select(ast, params),
}


def build_call(
    client: MilvusClient | AsyncMilvusClient,
    ast: exp.Expression,
    parameters: dict[str, t.Any],
) -> Call:
    """The one entry point both ``Cursor`` (sync) and ``aio`` (async)
    call: a parsed MilvusQL statement -> what to call on ``client``."""
    if isinstance(ast, exp.Create):
        if ast.args.get("kind") == "TABLE":
            return _build_create_table(client, ast, parameters)
        if ast.args.get("kind") == "INDEX":
            return _build_create_index(client, ast, parameters)
        msg = f"unsupported CREATE kind: {ast.args.get('kind')}"
        raise errors.NotSupportedError(msg)
    if isinstance(ast, exp.Alter):
        return _build_alter_add_field(ast)
    if isinstance(ast, exp.Drop):
        if ast.args.get("kind") == "TABLE":
            return _build_drop_table(ast)
        msg = f"unsupported DROP kind: {ast.args.get('kind')}"
        raise errors.NotSupportedError(msg)
    builder = _BUILDERS.get(type(ast))
    if builder is None:
        msg = f"unsupported statement: {ast.__class__.__name__}"
        raise errors.NotSupportedError(msg)
    return builder(client, ast, parameters)


__all__ = ["Call", "RowsAndDescription", "build_batch_call", "build_call"]
