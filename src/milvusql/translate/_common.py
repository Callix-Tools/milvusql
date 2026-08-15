"""Primitives shared by every translator in this package.

:mod:`ast_to_pymilvus` (one statement -> one ``pymilvus`` call) and
:mod:`relational` (JOIN/GROUP BY/subquery -> several calls plus a Polars
evaluation) both need the same handful of things: the :class:`Call`
shape a cursor knows how to run, the bind-value resolver, the ``WHERE``
-> Milvus-filter renderer and Milvus's own per-call row ceiling. They
live here rather than in either translator so neither has to import the
other -- ``ast_to_pymilvus`` dispatches *into* ``relational``, and
``relational`` reuses these primitives, which would otherwise be an
import cycle.

Nothing here performs I/O, same invariant the rest of the package keeps.
"""

from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass, field

from sqlglot import exp

from milvusql.dbapi import errors

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
DEFAULT_QUERY_LIMIT = 16384

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

    ``then`` exists for the statements that genuinely cannot be a single
    RPC. ``UPDATE`` is one: Milvus has no partial-row update, only
    ``upsert()``, which replaces the whole entity -- so an ``UPDATE ...
    SET ... WHERE ...`` has to read the matching rows in full first,
    merge the ``SET`` values in Python, then upsert the merged rows
    back. A ``JOIN`` (or any query reading more than one collection) is
    the other: :mod:`relational` chains one fetch per collection through
    exactly this hook. Given this call's raw result, ``then`` returns
    the next ``Call`` to run, or ``None`` to stop and postprocess *this*
    call's raw result instead -- still just describing what to call
    next, the same "no I/O in this module" invariant every builder
    keeps; ``dbapi/cursor.py`` (sync) and ``aio.py`` (async) are still
    the only two places that actually invoke a client method, in a small
    loop rather than once."""

    method: str
    kwargs: dict[str, t.Any] = field(default_factory=dict)
    postprocess: Postprocess = lambda _raw: ([], None, -1, None)
    then: t.Callable[[t.Any], Call | None] | None = None


def resolve_value(node: exp.Expression, parameters: dict[str, t.Any]) -> t.Any:  # noqa: ANN401 -- a bind value is genuinely any JSON-ish type
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
        return [resolve_value(e, parameters) for e in node.expressions]
    if isinstance(node, exp.Null):
        return None
    msg = f"unsupported value expression: {node.__class__.__name__}"
    raise errors.NotSupportedError(msg)


COMPARISON_OPS: dict[type[exp.Expression], str] = {
    exp.EQ: "==",
    exp.NEQ: "!=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
}


def render_filter_value(value: t.Any) -> str:  # noqa: ANN401
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


def render_filter(  # noqa: PLR0911, PLR0912 -- one return per AST node case, clearer flat than nested
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
    inline here, via :func:`render_filter_value`, is the fallback that
    is verified to actually work."""
    if isinstance(node, exp.Where):
        return render_filter(node.this, parameters)
    if isinstance(node, exp.Paren):
        return f"({render_filter(node.this, parameters)})"
    if isinstance(node, exp.And):
        left = render_filter(node.this, parameters)
        right = render_filter(node.expression, parameters)
        return f"({left} and {right})"
    if isinstance(node, exp.Or):
        left = render_filter(node.this, parameters)
        right = render_filter(node.expression, parameters)
        return f"({left} or {right})"
    if isinstance(node, exp.Not):
        return f"not ({render_filter(node.this, parameters)})"
    op = COMPARISON_OPS.get(type(node))
    if op is not None:
        left = render_filter(node.this, parameters)
        right = render_filter(node.expression, parameters)
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
        column = render_filter(node.this, parameters)
        values = ", ".join(
            render_filter_value(resolve_value(value, parameters))
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
        column = render_filter(node.this, parameters)
        pattern = render_filter_value(
            resolve_value(node.expression, parameters)
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
        column = render_filter(node.this, parameters)
        low = render_filter_value(resolve_value(node.args["low"], parameters))
        high = render_filter_value(
            resolve_value(node.args["high"], parameters)
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
        column = render_filter(node.this, parameters)
        return f"{column} is null"
    if isinstance(node, exp.Column):
        return node.this.name
    if isinstance(
        node, (exp.Placeholder, exp.Parameter, exp.Literal, exp.Boolean)
    ):
        return render_filter_value(resolve_value(node, parameters))
    msg = f"unsupported filter expression: {node.__class__.__name__}"
    raise errors.NotSupportedError(msg)


def filter_text(where: exp.Where | None, parameters: dict[str, t.Any]) -> str:
    if where is None:
        return ""
    return render_filter(where, parameters)


def description(output_names: list[str]) -> list[tuple]:
    return [
        (name, None, None, None, None, None, True) for name in output_names
    ]


__all__ = [
    "COMPARISON_OPS",
    "DEFAULT_QUERY_LIMIT",
    "Call",
    "Postprocess",
    "RowsAndDescription",
    "description",
    "filter_text",
    "render_filter",
    "render_filter_value",
    "resolve_value",
]
