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

import typing as t

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    RRFRanker,
    WeightedRanker,
)
from sqlglot import exp
from sqlglot_milvus.expressions import (
    CONSISTENCY_ARG,
    HYBRID_ARG,
    SEARCH_PARAMS_ARG,
    AddField,
    HybridSearch,
    LoadTable,
    ReleaseTable,
)

from milvusql.dbapi import errors
from milvusql.translate._common import (
    DEFAULT_QUERY_LIMIT,
    Call,
    Postprocess,
    Rows,
    RowsAndDescription,
    ann_metric,
    description,
    filter_text,
    resolve_value,
    unbounded_query_call,
)
from milvusql.translate.relational import (
    build_relational_call,
    build_set_operation_call,
    needs_relational_engine,
)

if t.TYPE_CHECKING:
    from pymilvus import AsyncMilvusClient, MilvusClient


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


def _update_zero(_rows: t.Any) -> RowsAndDescription:  # noqa: ANN401
    """``UPDATE`` ends here when nothing matched the ``WHERE`` clause --
    ``rowcount`` is honestly ``0``, not the ``-1`` "not applicable" DDL
    convention, since an UPDATE that matched nothing is a completed,
    well-understood outcome."""
    return [], None, 0, None


# -----------------------------------------------------------------------
# Shared value/property helpers
# -----------------------------------------------------------------------


def _property_list_to_dict(
    properties: list[exp.Property], parameters: dict[str, t.Any]
) -> dict[str, t.Any]:
    return {
        prop.this.name: resolve_value(prop.args["value"], parameters)
        for prop in properties
    }


def _properties_to_dict(
    node: exp.Properties | None, parameters: dict[str, t.Any]
) -> dict[str, t.Any]:
    if node is None:
        return {}
    return _property_list_to_dict(node.expressions, parameters)


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

#: The dimensioned vector spellings that (like ``SPARSEVEC``) parse as
#: a user-defined type carrying its own name: ``BINARYVEC(128)`` and
#: the reduced-precision families. Values are inserted as whatever
#: ``pymilvus`` accepts for the type (``bytes`` for binary, a numpy
#: array of the matching dtype for fp16/bf16/int8) -- the DBAPI passes
#: bind values through untouched, so no conversion happens here.
_USERDEFINED_VECTORS: dict[str, DataType] = {
    "BINARYVEC": DataType.BINARY_VECTOR,
    "FLOAT16VEC": DataType.FLOAT16_VECTOR,
    "BFLOAT16VEC": DataType.BFLOAT16_VECTOR,
    "INT8VEC": DataType.INT8_VECTOR,
}

#: Every Milvus vector field type. Used for the one-vector-per-table
#: check and for "a vector field can't be nullable".
_VECTOR_TYPES = frozenset(
    {
        DataType.FLOAT_VECTOR,
        DataType.SPARSE_FLOAT_VECTOR,
        *_USERDEFINED_VECTORS.values(),
    }
)

#: ``ARRAY<T>(n)`` with no explicit capacity: Milvus requires
#: ``max_capacity`` on every array field, so a spelling without one
#: gets the type's own documented maximum rather than a rejection.
_DEFAULT_ARRAY_CAPACITY = 4096


def _map_datatype(  # noqa: PLR0911 -- one return per type family, clearer flat than nested
    dtype: exp.DataType,
) -> tuple[DataType, dict[str, t.Any]]:
    """A parsed ``ColumnDef.kind`` -> ``(pymilvus DataType, extra
    add_field kwargs)``. Anything not mapped is a real gap, not a
    silent guess, so it raises rather than picking an arbitrary Milvus
    type for a spelling this layer has never seen used.
    """
    if dtype.this is exp.DataType.Type.VARCHAR:
        param = dtype.expressions[0] if dtype.expressions else None
        max_length = int(param.this.this) if param is not None else 65535
        return DataType.VARCHAR, {"max_length": max_length}
    if dtype.this is exp.DataType.Type.TEXT:
        # `TEXT` is the full-text column: max-length VARCHAR with the
        # analyzer on (so a BM25 function can consume it) and keyword
        # match on (so `MATCH ... AGAINST` filters work) -- the two
        # flags Milvus's own full-text walkthroughs always set together.
        return DataType.VARCHAR, {
            "max_length": 65535,
            "enable_analyzer": True,
            "enable_match": True,
        }
    if dtype.this is exp.DataType.Type.ARRAY:
        if not dtype.expressions or not isinstance(
            dtype.expressions[0], exp.DataType
        ):
            msg = "ARRAY needs an element type: ARRAY<BIGINT>(100)"
            raise errors.NotSupportedError(msg)
        element_type, element_extra = _map_datatype(dtype.expressions[0])
        if element_type is DataType.ARRAY or element_type in _VECTOR_TYPES:
            msg = (
                "unsupported ARRAY element type: "
                f"{dtype.expressions[0].sql(dialect='milvus')} -- Milvus "
                "arrays hold scalars only"
            )
            raise errors.NotSupportedError(msg)
        values = dtype.args.get("values") or []
        capacity = int(values[0].this) if values else _DEFAULT_ARRAY_CAPACITY
        return DataType.ARRAY, {
            "element_type": element_type,
            "max_capacity": capacity,
            # A VARCHAR element's max_length rides along -- pymilvus
            # takes it as a sibling kwarg on the array field itself.
            **element_extra,
        }
    if dtype.this is exp.DataType.Type.VECTOR:
        dim = int(dtype.expressions[0].this.this)
        return DataType.FLOAT_VECTOR, {"dim": dim}
    if dtype.this is exp.DataType.Type.USERDEFINED:
        # ``SPARSEVEC`` and the dimensioned vector spellings have no
        # dedicated sqlglot ``DataType.Type`` of their own (unlike
        # ``VECTOR``, recycled from a real one) -- they parse as a
        # generic user-defined type carrying their own spelling as free
        # text in ``kind`` (confirmed directly against sqlglot's own
        # parser output), case-preserved from whatever the caller
        # wrote, hence the case-insensitive matches.
        kind = dtype.args.get("kind")
        kind_name = kind.upper() if isinstance(kind, str) else ""
        if kind_name == "SPARSEVEC":
            return DataType.SPARSE_FLOAT_VECTOR, {}
        dimensioned = _USERDEFINED_VECTORS.get(kind_name)
        if dimensioned is not None:
            if not dtype.expressions:
                msg = f"{kind_name} needs a dimension: {kind_name}(128)"
                raise errors.NotSupportedError(msg)
            return dimensioned, {"dim": int(dtype.expressions[0].this.this)}
    mapped = _SCALAR_TYPES.get(dtype.this)
    if mapped is None:
        msg = f"unsupported column type: {dtype.sql(dialect='milvus')}"
        raise errors.NotSupportedError(msg)
    return mapped, {}


def _generated_bm25_input(column: exp.ColumnDef) -> str | None:
    """``GENERATED ALWAYS AS (BM25(content))`` / ``AS (BM25(content))``
    on a column -> the input column's name, or ``None`` when the column
    is not generated.

    Both spellings parse (confirmed directly): the long form to a
    ``GeneratedAsIdentityColumnConstraint`` carrying the expression, the
    short form to a ``ComputedColumnConstraint`` wrapping it in a
    ``Paren``. ``BM25`` is the only generator Milvus computes
    server-side, so anything else is rejected by name."""
    for constraint in column.constraints or []:
        kind = constraint.kind
        if isinstance(kind, exp.ComputedColumnConstraint):
            expression = kind.this
        elif isinstance(kind, exp.GeneratedAsIdentityColumnConstraint):
            expression = kind.args.get("expression")
            if expression is None:
                # `GENERATED ALWAYS AS IDENTITY` -- an identity column,
                # not a computed one; the AUTO_INCREMENT handling owns it.
                continue
        else:
            continue
        while isinstance(expression, exp.Paren):
            expression = expression.this
        if (
            isinstance(expression, exp.Anonymous)
            and str(expression.this).upper() == "BM25"
            and len(expression.expressions) == 1
            and isinstance(expression.expressions[0], exp.Column)
        ):
            return expression.expressions[0].name
        rendered = (
            expression.sql(dialect="milvus")
            if isinstance(expression, exp.Expression)
            else str(expression)
        )
        msg = (
            f"unsupported generated column expression: {rendered!r} -- "
            "BM25(<text column>) is the one generator Milvus computes "
            "server-side"
        )
        raise errors.NotSupportedError(msg)
    return None


def _build_create_table(
    client: MilvusClient | AsyncMilvusClient,
    ast: exp.Create,
    parameters: dict[str, t.Any],
) -> Call:
    schema_node = ast.this
    table_name = schema_node.this.name
    props = _properties_to_dict(ast.args.get("properties"), parameters)

    columns = [
        c for c in schema_node.expressions if isinstance(c, exp.ColumnDef)
    ]
    if not any(_map_datatype(c.kind)[0] in _VECTOR_TYPES for c in columns):
        # Milvus refuses `create_collection()` outright for a schema
        # with zero vector fields -- confirmed directly against a
        # real, non-Lite server (Milvus Lite silently tolerates it,
        # which is what let this go unnoticed for a while). Raised
        # here, client-side, before `create_schema()`/any RPC even
        # runs: the server's own rejection is a bare, unhelpful
        # low-level gRPC error with no indication of *why*. This is a
        # real, documented limitation, not a bug to paper over --
        # notably, it means a tool that creates a table with no vector
        # column of its own (e.g. Alembic's ``alembic_version``
        # bookkeeping table) cannot be made to work against this
        # backend.
        msg = (
            f"CREATE TABLE {table_name!r} has no VECTOR/SPARSEVEC "
            "column -- Milvus requires at least one vector field per "
            "collection."
        )
        raise errors.NotSupportedError(msg)

    milvus_schema = client.create_schema(
        enable_dynamic_field=False,
        partition_key_field=props.pop("partition_key", None),
    )
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
        generator_source = _generated_bm25_input(column)
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
            and milvus_type not in _VECTOR_TYPES
        )
        milvus_schema.add_field(
            field_name=name,
            datatype=milvus_type,
            is_primary=is_primary,
            auto_id=exp.AutoIncrementColumnConstraint in constraint_kinds,
            nullable=nullable,
            **extra,
        )
        if generator_source is not None:
            # `content_sparse SPARSEVEC GENERATED ALWAYS AS
            # (BM25(content))` -- the field itself plus the schema-level
            # BM25 function Milvus computes it with, which is the whole
            # full-text pipeline: the server tokenizes `content` (a
            # `TEXT` column, i.e. analyzer-enabled VARCHAR) into this
            # sparse field, and a search against it with
            # `metric_type='BM25'` ranks by relevance.
            if milvus_type is not DataType.SPARSE_FLOAT_VECTOR:
                msg = (
                    f"column {name!r}: BM25 generates a SPARSEVEC, not "
                    f"{column.kind.sql(dialect='milvus')}"
                )
                raise errors.NotSupportedError(msg)
            milvus_schema.add_function(
                Function(
                    name=f"{name}_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=[generator_source],
                    output_field_names=[name],
                )
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
    """``ALTER TABLE items ADD FIELD tag VARCHAR(32)`` or the standard
    SQL spelling ``ALTER TABLE items ADD COLUMN tag VARCHAR(32)``.

    sqlglot-milvus's own grammar rejects every other ``ALTER`` action
    (``DROP``/``ALTER COLUMN``/``MODIFY``) at *parse* time with a
    ``ParseError`` (Milvus can't perform them at all), so the single
    action reaching here is always either an ``AddField`` (Milvus's own
    ``ADD FIELD`` spelling, wrapping a ``ColumnDef``) or a bare
    ``ColumnDef`` (standard SQL's ``ADD COLUMN`` spelling, e.g. as
    rendered by Alembic's default ``add_column()``) -- both mean the
    same thing and are unwrapped to the same ``ColumnDef`` below. The
    ``else`` branch is defense in depth against a synthetic or
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
    if len(actions) != 1:
        msg = "unsupported ALTER TABLE action"
        raise errors.NotSupportedError(msg)
    action = actions[0]
    if isinstance(action, AddField):
        column = action.this
    elif isinstance(action, exp.ColumnDef):
        column = action
    else:
        msg = "unsupported ALTER TABLE action"
        raise errors.NotSupportedError(msg)
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
            name: resolve_value(value, parameters)
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
    filter_expr = filter_text(ast.args.get("where"), parameters)
    kwargs: dict[str, t.Any] = {"collection_name": table_name}
    if filter_expr:
        kwargs["filter"] = filter_expr
    return Call("delete", kwargs, _mutation_count("delete_count"))


def _build_update(ast: exp.Update, parameters: dict[str, t.Any]) -> Call:
    """``UPDATE table SET col = val[, ...] WHERE ...`` -- see
    :class:`Call`'s docstring for why this is a read (``query``) whose
    ``then`` chains into a write (``upsert``), not one RPC.

    Known, confirmed limitation for an ``auto_id`` primary key (every
    table this module's own ``AUTO_INCREMENT`` DDL creates): a real
    (non-Lite) Milvus server's ``upsert()`` does **not** honor the
    primary key value in ``merged`` below for an ``auto_id=True``
    collection -- it silently mints a *new* server-assigned id for the
    row regardless, even though the row's other columns are correctly
    overwritten in place (confirmed directly: updating only a
    non-primary-key column still changes the row's id; the total row
    count is unaffected, so this is a rename in place, not a
    duplicate). Milvus Lite was more lenient here and preserved the
    given id, which is what let this go unnoticed until tested against
    a real server. There is no workaround through any Milvus API: an
    ``auto_id`` field categorically cannot be given an explicit value
    on any write, upsert included, so an ``UPDATE`` can never guarantee
    the row's id survives it. Callers that need a stable identity
    across updates need a non-``auto_id`` (caller-assigned) primary
    key instead."""
    table_name = ast.this.name
    # `UPDATE`'s `SET` list is always `col = <expr>` (`exp.EQ`) by
    # grammar -- `resolve_value` is what actually rejects an
    # unsupported right-hand side (`SET stock = stock + 1`, an `F()`
    # expression in Django terms: `exp.Add`, not a bare bind value or
    # literal), the same way it rejects one anywhere else in this
    # module.
    set_values = {
        assignment.this.name: resolve_value(assignment.expression, parameters)
        for assignment in ast.expressions
    }
    filter_expr = filter_text(ast.args.get("where"), parameters)

    query_kwargs: dict[str, t.Any] = {
        "collection_name": table_name,
        # Every field, vectors included -- `upsert()` replaces the
        # whole entity, so the merged row has to carry everything the
        # row already had, not just the columns SET touches.
        "output_fields": ["*"],
    }
    if filter_expr:
        query_kwargs["filter"] = filter_expr

    def _write_back(rows: Rows) -> Call | None:
        if not rows:
            return None
        merged = [{**row, **set_values} for row in rows]
        return _chunked_upsert(table_name, merged)

    # The read pages past Milvus's per-call row ceiling (it used to
    # raise there), so a broad UPDATE now reads everything it matches
    # and writes it back in bounded chunks.
    return unbounded_query_call(query_kwargs, _write_back, _update_zero)


#: How many merged rows one ``upsert`` RPC carries. Milvus itself takes
#: arbitrarily long lists, but the gRPC message has a size cap and every
#: row carries its full vector here -- the same order of magnitude
#: pymilvus's own bulk helpers batch at.
_UPSERT_BATCH_ROWS = 1000


def _chunked_upsert(table_name: str, merged: Rows) -> Call:
    """The write half of ``UPDATE``: every merged row, ``upsert``\\ ed in
    :data:`_UPSERT_BATCH_ROWS`-row chunks chained through ``then``, with
    one total ``rowcount`` at the end."""
    total = {"count": 0}

    def link(start: int) -> Call:
        chunk = merged[start : start + _UPSERT_BATCH_ROWS]

        def then(raw: t.Any) -> Call | None:  # noqa: ANN401
            if isinstance(raw, dict):
                total["count"] += raw.get("upsert_count", 0)
            following = start + _UPSERT_BATCH_ROWS
            return link(following) if following < len(merged) else None

        def postprocess(_raw: t.Any) -> RowsAndDescription:  # noqa: ANN401
            return [], None, total["count"], None

        return Call(
            "upsert",
            {"collection_name": table_name, "data": chunk},
            postprocess,
            then=then,
        )

    return link(0)


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
        return rows, description(output_names), len(rows), None

    return postprocess


def _query_rows(
    field_names: list[str], output_names: list[str]
) -> Postprocess:
    def postprocess(raw: list[dict[str, t.Any]]) -> RowsAndDescription:
        rows = [tuple(row.get(name) for name in field_names) for row in raw]
        return rows, description(output_names), len(rows), None

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
    every matching row in Python and only *then* applies the real
    ``LIMIT`` -- sorting after truncating would return the wrong rows
    entirely. The fetch behind it pages past Milvus's per-call row
    ceiling (see :func:`unbounded_query_call`), so the sort really does
    run over every matching row."""

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
        return tuples, description(output_names), len(tuples), None

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


def _flatten_trivial_subquery(ast: exp.Select) -> exp.Select:
    """Unwrap a ``FROM (SELECT ... FROM <table> [WHERE ...]) AS x``
    that does nothing but relabel columns and (optionally) filter a
    single real table -- exactly the shape SQLAlchemy's legacy
    ``Query.count()``/``Query.exists()`` wrap *every* query in
    (confirmed directly: ``session.query(Model).filter(...).count()``
    puts the filter inside the subquery, not the outer one). Rejecting
    every subquery outright -- correct for a genuine join, union or
    aggregate subquery -- would also break that common, harmless idiom.

    Safe because the inner and outer ``WHERE`` both restrict the same
    rows of the same table, so they can be ANDed together against the
    real table with no change in meaning. Only fires when the inner
    query has no ``JOIN``/``GROUP BY``/``ORDER BY``/``LIMIT``/
    ``DISTINCT`` of its own -- any of those would change which rows or
    how many reach the outer query, and this function does not attempt
    to merge that. Returns ``ast`` unchanged when there is nothing this
    confident to flatten (including a bare ``SELECT`` with no ``FROM``
    at all, e.g. ``SELECT EXISTS(...)`` -- :func:`_table_name` still
    rejects that cleanly, same as every other shape left unflattened)."""
    if ast.args.get("group") is not None or ast.args.get("joins"):
        # The relational engine handles this statement, and it addresses
        # the subquery by its alias (`t.category`) -- rewriting the FROM
        # out from under those references would leave them pointing at a
        # name that no longer exists. Nothing is lost by not flattening:
        # that engine plans the subquery's own filter into its scan
        # anyway, which is all this rewrite buys.
        return ast
    from_node = ast.args.get("from_")
    if from_node is None:
        return ast
    from_this = from_node.this
    if not isinstance(from_this, exp.Subquery):
        return ast
    inner = from_this.this
    inner_from = (
        inner.args.get("from_") if isinstance(inner, exp.Select) else None
    )
    if (
        not isinstance(inner, exp.Select)
        or not isinstance(inner_from, exp.From)
        or not isinstance(inner_from.this, exp.Table)
        or inner.args.get("joins")
        or inner.args.get("group")
        or inner.args.get("order")
        or inner.args.get("limit")
        or inner.args.get("distinct")
    ):
        return ast
    flattened = ast.copy()
    flattened.set("from_", inner_from.copy())
    outer_where = ast.args.get("where")
    inner_where = inner.args.get("where")
    if inner_where is not None:
        combined = (
            exp.and_(outer_where.this, inner_where.this)
            if outer_where is not None
            else inner_where.this
        )
        flattened.set("where", exp.Where(this=combined))
    return flattened


def _table_name(ast: exp.Select) -> str:
    """The single collection a ``SELECT`` reads from.

    Milvus has no cross-collection JOIN and no subquery-as-source, so
    both are rejected explicitly here rather than silently narrowing to
    whatever ``.this.name`` happens to resolve to: a ``JOIN`` was
    previously dropped without a word (the query just ran against the
    first table, as if the ``JOIN`` had never been written), and a
    subquery in ``FROM`` produced an unclear ``AttributeError`` deep in
    ``pymilvus`` instead of a clean, actionable error at translate
    time. ``_flatten_trivial_subquery`` (called before this, in
    ``_build_select``) already unwraps the one shape of subquery that
    is safe to allow through, so anything still a ``Subquery`` here is
    genuinely unsupported."""
    if ast.args.get("joins"):
        msg = "JOIN is not supported: Milvus has no cross-collection join"
        raise errors.NotSupportedError(msg)
    from_node = ast.args.get("from_")
    if from_node is None:
        # A bare `SELECT EXISTS(...)`/`SELECT 1` with no FROM at all --
        # every collection-reading builder needs one. Raising cleanly
        # here beats the raw `KeyError` a bare `ast.args["from_"]`
        # would otherwise surface.
        msg = "SELECT with no FROM clause is not supported"
        raise errors.NotSupportedError(msg)
    from_this = from_node.this
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
        return [row], description(output_names), 1, None

    return postprocess


def _reduce_aggregate_rows(
    specs: list[tuple[str, str | None]], output_names: list[str]
) -> Postprocess:
    """``SUM``/``AVG``/``MIN``/``MAX``/``COUNT(<column>)`` (everything
    but a pure ``COUNT(*)``, which the server-side fast path above
    handles instead): Milvus computes none of these itself, so every
    matching row is fetched -- paging past the per-call row ceiling
    where needed (see :func:`unbounded_query_call`) -- and reduced here
    in Python, over the complete set of matching rows."""

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
        return [tuple(values)], description(output_names), 1, None

    return postprocess


def _build_aggregate(ast: exp.Select, parameters: dict[str, t.Any]) -> Call:
    table_name = _table_name(ast)
    filter_expr = filter_text(ast.args.get("where"), parameters)
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
        # past `DEFAULT_QUERY_LIMIT` rows).
        kwargs: dict[str, t.Any] = {
            "collection_name": table_name,
            "output_fields": ["count(*)"],
        }
        if filter_expr:
            kwargs["filter"] = filter_expr
        return Call(
            "query", kwargs, _count_star_rows(len(specs), output_names)
        )

    columns = sorted({column for _, column in specs if column is not None})
    kwargs = {
        "collection_name": table_name,
        "output_fields": columns or ["count(*)"],
    }
    if filter_expr:
        kwargs["filter"] = filter_expr
    return unbounded_query_call(
        kwargs, None, _reduce_aggregate_rows(specs, output_names)
    )


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
    filter_expr = filter_text(ast.args.get("where"), parameters)
    limit_node = ast.args.get("limit")
    limit = (
        int(resolve_value(limit_node.expression, parameters))
        if limit_node
        else DEFAULT_QUERY_LIMIT
    )

    reqs = []
    weights = []
    for arm in hybrid.expressions:
        distance_node = arm.this
        metric_type = ann_metric(distance_node)
        if metric_type is None:
            msg = (
                "unsupported HYBRID SEARCH arm scoring expression: "
                f"{distance_node.__class__.__name__}"
            )
            raise errors.NotSupportedError(msg)
        column_name = distance_node.this.this.name
        query_vector = resolve_value(distance_node.expression, parameters)
        weight_node = arm.args.get("weight")
        weight = (
            float(resolve_value(weight_node, parameters))
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
        if filter_expr:
            req_kwargs["expr"] = filter_expr
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


def _build_select(  # noqa: PLR0911, PLR0912 -- one return per SELECT shape, clearer flat than nested
    ast: exp.Select, parameters: dict[str, t.Any]
) -> Call:
    ast = _flatten_trivial_subquery(ast)
    hybrid = ast.args.get(HYBRID_ARG)
    if hybrid:
        return _build_hybrid_search(hybrid, ast, parameters)

    if needs_relational_engine(ast):
        # A JOIN, a GROUP BY or a subquery: more than one collection
        # read, or a reduction Milvus has no RPC for. `relational` plans
        # those into one fetch per collection plus a client-side
        # evaluation -- everything else stays exactly one RPC and never
        # touches that path. Checked before `_is_aggregate_select`
        # below, which looks only at the SELECT list and would otherwise
        # claim `SELECT COUNT(*) FROM a JOIN b` as a single-collection
        # aggregate; the bare `SELECT COUNT(*) FROM items` it is for
        # still lands there, since one collection with no GROUP BY never
        # needs this engine.
        return build_relational_call(ast, parameters)

    if _is_aggregate_select(ast):
        return _build_aggregate(ast, parameters)

    table_name = _table_name(ast)
    field_names = _select_field_names(ast)
    output_names = _select_output_names(ast)
    filter_expr = filter_text(ast.args.get("where"), parameters)
    limit_node = ast.args.get("limit")
    # LIMIT is a literal in hand-written MilvusQL, but a bound
    # parameter in SQLAlchemy-generated text by default (confirmed
    # directly: `.limit(5)` compiles to `LIMIT :param_1`, not
    # `LIMIT 5`) -- resolve through the same value path as everything
    # else instead of assuming a literal. No LIMIT clause at all (a
    # bare `Model.objects.all()`) falls back to Milvus's own per-call
    # ceiling, never an arbitrary smaller number -- see
    # `DEFAULT_QUERY_LIMIT`.
    limit = (
        int(resolve_value(limit_node.expression, parameters))
        if limit_node
        else DEFAULT_QUERY_LIMIT
    )

    order = ast.args.get("order")
    if order is not None:
        ordered = order.expressions[0]
        distance_node = ordered.this
        metric_type = ann_metric(distance_node)
    else:
        metric_type = None

    if order is None or metric_type is None:
        kwargs: dict[str, t.Any] = {
            "collection_name": table_name,
            "output_fields": field_names,
        }
        if filter_expr:
            kwargs["filter"] = filter_expr
        if order is None:
            if limit_node is not None:
                # A bounded read stays exactly one RPC.
                kwargs["limit"] = limit
                return Call(
                    "query", kwargs, _query_rows(field_names, output_names)
                )
            # No LIMIT clause at all (a bare `Model.objects.all()`):
            # fetch everything, paging past Milvus's per-call row
            # ceiling where the result demands it.
            return unbounded_query_call(
                kwargs, None, _query_rows(field_names, output_names)
            )
        # A plain scalar `ORDER BY` -- not a distance operator, so this
        # isn't a vector search. Fetch every matching row (paging past
        # the ceiling where needed) and sort client-side instead of
        # rejecting it outright.
        order_keys = _select_order_keys(order, field_names)
        fetch_fields = list(field_names)
        for column, _ in order_keys:
            if column not in fetch_fields:
                fetch_fields.append(column)
        kwargs["output_fields"] = fetch_fields
        return unbounded_query_call(
            kwargs,
            None,
            _sorted_query_rows(field_names, output_names, order_keys, limit),
        )

    column_name = distance_node.this.this.name
    query_vector = resolve_value(distance_node.expression, parameters)

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
    if filter_expr:
        kwargs["filter"] = filter_expr
    consistency = ast.args.get(CONSISTENCY_ARG)
    if consistency is not None:
        kwargs["consistency_level"] = consistency.this.name

    return Call("search", kwargs, _search_rows(field_names, output_names))


# -----------------------------------------------------------------------
# Introspection: SHOW TABLES / SHOW DATABASES / DESCRIBE
# -----------------------------------------------------------------------

#: pymilvus ``DataType`` -> the MilvusQL spelling ``DESCRIBE`` prints,
#: so its output round-trips into ``CREATE TABLE`` unchanged.
_DESCRIBE_TYPE_NAMES: dict[DataType, str] = {
    DataType.INT8: "TINYINT",
    DataType.INT16: "SMALLINT",
    DataType.INT32: "INT",
    DataType.INT64: "BIGINT",
    DataType.FLOAT: "FLOAT",
    DataType.DOUBLE: "DOUBLE",
    DataType.BOOL: "BOOLEAN",
    DataType.JSON: "JSON",
    DataType.VARCHAR: "VARCHAR",
    DataType.FLOAT_VECTOR: "VECTOR",
    DataType.SPARSE_FLOAT_VECTOR: "SPARSEVEC",
    DataType.BINARY_VECTOR: "BINARYVEC",
    DataType.FLOAT16_VECTOR: "FLOAT16VEC",
    DataType.BFLOAT16_VECTOR: "BFLOAT16VEC",
    DataType.INT8_VECTOR: "INT8VEC",
}


def _describe_type_text(field: dict[str, t.Any]) -> str:
    """One ``describe_collection`` field -> its MilvusQL type spelling
    (``VARCHAR(64)``, ``VECTOR(8)``, ``ARRAY<BIGINT>(100)``)."""
    datatype = DataType(field["type"])
    params = field.get("params") or {}
    if datatype is DataType.ARRAY:
        element = field.get("element_type", params.get("element_type"))
        element_name = (
            _DESCRIBE_TYPE_NAMES.get(DataType(element), str(element))
            if element is not None
            else "?"
        )
        capacity = params.get("max_capacity")
        suffix = f"({capacity})" if capacity else ""
        return f"ARRAY<{element_name}>{suffix}"
    name = _DESCRIBE_TYPE_NAMES.get(datatype, datatype.name)
    size = params.get("dim") or params.get("max_length")
    if size and datatype is not DataType.SPARSE_FLOAT_VECTOR:
        return f"{name}({size})"
    return name


_DESCRIBE_COLUMNS = ["Field", "Type", "Null", "Key", "Extra"]


def _describe_rows(raw: dict[str, t.Any]) -> RowsAndDescription:
    rows = []
    for field in raw.get("fields", []):
        extras = []
        if field.get("auto_id"):
            extras.append("auto_increment")
        for function in raw.get("functions", []):
            if field["name"] in (function.get("output_field_names") or []):
                inputs = ", ".join(function.get("input_field_names") or [])
                extras.append(f"generated by BM25({inputs})")
        rows.append(
            (
                field["name"],
                _describe_type_text(field),
                "YES" if field.get("nullable") else "NO",
                "PRI" if field.get("is_primary") else "",
                ", ".join(extras),
            )
        )
    return rows, description(_DESCRIBE_COLUMNS), len(rows), None


def _build_describe(ast: exp.Describe) -> Call:
    return Call(
        "describe_collection",
        {"collection_name": ast.this.name},
        _describe_rows,
    )


def _name_list_rows(label: str) -> Postprocess:
    def postprocess(raw: list[str]) -> RowsAndDescription:
        rows = [(name,) for name in raw]
        return rows, description([label]), len(rows), None

    return postprocess


def _build_show(ast: exp.Command) -> Call:
    """``SHOW ...`` reaches this layer as an opaque ``exp.Command`` (the
    base grammar has no structured SHOW), so the one word after ``SHOW``
    is read here: ``TABLES`` and ``DATABASES`` are the two Milvus can
    answer."""
    argument = ast.args.get("expression")
    text = (argument.this if isinstance(argument, exp.Literal) else "").strip()
    keyword = text.split()[0].upper() if text.split() else ""
    if keyword == "TABLES":
        return Call("list_collections", {}, _name_list_rows("table_name"))
    if keyword == "DATABASES":
        return Call("list_databases", {}, _name_list_rows("database_name"))
    msg = f"unsupported SHOW statement: SHOW {text or '?'}"
    raise errors.NotSupportedError(msg)


# -----------------------------------------------------------------------
# Databases and indexes
# -----------------------------------------------------------------------


def _build_drop_index(ast: exp.Drop) -> Call:
    on = ast.args.get("cluster")
    if on is None:
        # The grammar parses a bare `DROP INDEX idx`, but Milvus scopes
        # index names to a collection, so the statement is unanswerable
        # without one.
        msg = "DROP INDEX needs the collection: DROP INDEX idx ON items"
        raise errors.ProgrammingError(msg)
    return Call(
        "drop_index",
        {"collection_name": on.this.name, "index_name": ast.this.name},
        _no_rows,
    )


def _build_use(ast: exp.Use) -> Call:
    return Call("use_database", {"db_name": ast.this.name}, _no_rows)


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
    # A set operation is not a `Select`, so it needs its own entries:
    # each branch is planned like any other query, into the same chain,
    # and the rows are combined once both are in hand.
    exp.Union: lambda _client, ast, params: build_set_operation_call(
        ast, params
    ),
    exp.Intersect: lambda _client, ast, params: build_set_operation_call(
        ast, params
    ),
    exp.Except: lambda _client, ast, params: build_set_operation_call(
        ast, params
    ),
    exp.Describe: lambda _client, ast, _params: _build_describe(ast),
    exp.Use: lambda _client, ast, _params: _build_use(ast),
}


def build_call(  # noqa: PLR0911 -- one return per statement kind, clearer flat than nested
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
        if ast.args.get("kind") in ("DATABASE", "SCHEMA"):
            return Call(
                "create_database", {"db_name": ast.this.name}, _no_rows
            )
        msg = f"unsupported CREATE kind: {ast.args.get('kind')}"
        raise errors.NotSupportedError(msg)
    if isinstance(ast, exp.Alter):
        return _build_alter_add_field(ast)
    if isinstance(ast, exp.Drop):
        if ast.args.get("kind") == "TABLE":
            return _build_drop_table(ast)
        if ast.args.get("kind") == "INDEX":
            return _build_drop_index(ast)
        if ast.args.get("kind") in ("DATABASE", "SCHEMA"):
            return Call("drop_database", {"db_name": ast.this.name}, _no_rows)
        msg = f"unsupported DROP kind: {ast.args.get('kind')}"
        raise errors.NotSupportedError(msg)
    if isinstance(ast, exp.Command) and str(ast.this).upper() == "SHOW":
        return _build_show(ast)
    builder = _BUILDERS.get(type(ast))
    if builder is None:
        msg = f"unsupported statement: {ast.__class__.__name__}"
        raise errors.NotSupportedError(msg)
    return builder(client, ast, parameters)


__all__ = ["Call", "RowsAndDescription", "build_batch_call", "build_call"]
