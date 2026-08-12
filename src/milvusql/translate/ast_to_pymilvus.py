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

from pymilvus import DataType
from sqlglot import exp
from sqlglot_milvus.expressions import (
    CONSISTENCY_ARG,
    HYBRID_ARG,
    METRIC_TYPES,
    SEARCH_PARAMS_ARG,
    LoadTable,
    ReleaseTable,
)

from milvusql.dbapi import errors

if t.TYPE_CHECKING:
    from pymilvus import AsyncMilvusClient, MilvusClient

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
    to turn its return value into DBAPI-shaped rows."""

    method: str
    kwargs: dict[str, t.Any] = field(default_factory=dict)
    postprocess: Postprocess = lambda _raw: ([], None, -1, None)


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


def _render_filter(  # noqa: PLR0911 -- one return per AST node case, clearer flat than nested
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


# ALTER TABLE ADD FIELD (Track A's ``AddField`` node) is intentionally
# not wired up here yet: ``MilvusClient.add_collection_field`` returns
# ``UNIMPLEMENTED`` against Milvus Lite (verified directly -- a raw
# ``grpc.RpcError``, not even a ``MilvusException``), so there is no way
# to test it in this phase's Milvus Lite-only suite. Out of the
# phase-1 scope committed in the design plan; add it back once there is
# a real (or newer Lite) server to verify against.


# -----------------------------------------------------------------------
# DML: INSERT / DELETE
# -----------------------------------------------------------------------


def _build_insert(ast: exp.Insert, parameters: dict[str, t.Any]) -> Call:
    schema_node = ast.this
    table_name = schema_node.this.name
    columns = [ident.name for ident in schema_node.expressions]
    values_node = ast.args["expression"]

    rows = [
        {
            name: _resolve_value(value, parameters)
            for name, value in zip(columns, row.expressions, strict=True)
        }
        for row in values_node.expressions
    ]
    return Call(
        "insert",
        {"collection_name": table_name, "data": rows},
        _insert_result,
    )


def _build_delete(ast: exp.Delete, parameters: dict[str, t.Any]) -> Call:
    table_name = ast.this.name
    filter_text = _filter_text(ast.args.get("where"), parameters)
    kwargs: dict[str, t.Any] = {"collection_name": table_name}
    if filter_text:
        kwargs["filter"] = filter_text
    return Call("delete", kwargs, _mutation_count("delete_count"))


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


def _build_select(ast: exp.Select, parameters: dict[str, t.Any]) -> Call:
    if ast.args.get(HYBRID_ARG):
        msg = "HYBRID SEARCH is not implemented yet (out of phase-1 scope)"
        raise errors.NotSupportedError(msg)

    table_name = ast.args["from_"].this.name
    field_names = _select_field_names(ast)
    output_names = _select_output_names(ast)
    filter_text = _filter_text(ast.args.get("where"), parameters)
    limit_node = ast.args.get("limit")
    # LIMIT is a literal in hand-written MilvusQL, but a bound
    # parameter in SQLAlchemy-generated text by default (confirmed
    # directly: `.limit(5)` compiles to `LIMIT :param_1`, not
    # `LIMIT 5`) -- resolve through the same value path as everything
    # else instead of assuming a literal.
    limit = (
        int(_resolve_value(limit_node.expression, parameters))
        if limit_node
        else 10
    )

    order = ast.args.get("order")
    if order is None:
        kwargs: dict[str, t.Any] = {
            "collection_name": table_name,
            "output_fields": field_names,
            "limit": limit,
        }
        if filter_text:
            kwargs["filter"] = filter_text
        return Call("query", kwargs, _query_rows(field_names, output_names))

    ordered = order.expressions[0]
    distance_node = ordered.this
    metric_type = METRIC_TYPES.get(type(distance_node))
    if metric_type is None:
        msg = "ORDER BY must be a distance operator for vector search"
        raise errors.NotSupportedError(msg)

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
        msg = "ALTER TABLE is not implemented yet (out of phase-1 scope)"
        raise errors.NotSupportedError(msg)
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


__all__ = ["Call", "RowsAndDescription", "build_call"]
