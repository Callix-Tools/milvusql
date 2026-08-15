"""JOIN, GROUP BY and subqueries: what Milvus itself cannot do.

Milvus reads one collection per RPC. It has no join, no grouped
aggregation and no subquery -- so the only way to answer
``SELECT ... FROM a JOIN b`` is to read from ``a``, read from ``b``, and
combine the two client-side. That is exactly what this module plans and
what Polars then executes.

The split of labour:

* **Milvus does what only Milvus can.** Every scalar predicate that
  mentions a single collection is pushed into that collection's own
  ``filter``; only the columns a query actually references are fetched;
  an ``ORDER BY <vector> <=> :q LIMIT k`` stays an ANN ``search`` (an
  index lookup here is the whole point -- re-ranking client-side would
  mean fetching the collection). Equi-join keys learned from an earlier
  scan are pushed into the later one as ``key in [...]``, so joining a
  50-row search result against a million-row collection reads 50 rows,
  not a million.
* **Polars does the relational algebra.** Joins, grouped aggregation,
  ``HAVING``, cross-collection predicates, ordering, ``DISTINCT``,
  ``LIMIT``/``OFFSET`` and the subquery evaluations run over the fetched
  frames.

Planning is pure: :func:`build_relational_call` returns a
:class:`~milvusql.translate._common.Call` chain, one link per scan, and
does no I/O -- ``dbapi/cursor.py`` and ``aio.py`` remain the only two
places that touch a client, so sync and async share this module verbatim
(D1/D12), exactly as they already share the single-RPC translator.

**Truncation is an error, never a silent wrong answer.** A join or a
grouped aggregate computed over *some* of the matching rows is a wrong
number with nothing to indicate it is wrong, so a scan that comes back
at Milvus's per-call ceiling raises instead of combining -- the same
choice, for the same reason, that the single-collection aggregate and
``ORDER BY`` paths in :mod:`ast_to_pymilvus` already make.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import polars as pl
from sqlglot import exp
from sqlglot_milvus.expressions import (
    CONSISTENCY_ARG,
    METRIC_TYPES,
    SEARCH_PARAMS_ARG,
)

from milvusql.dbapi import errors
from milvusql.translate._common import (
    COMPARISON_OPS,
    DEFAULT_QUERY_LIMIT,
    Call,
    RowsAndDescription,
    description,
    render_filter,
    render_filter_value,
    resolve_value,
)

#: How many distinct join-key values are worth pushing into a later
#: scan's ``filter`` as ``key in [...]``. Past this the filter string
#: itself becomes the problem (Milvus parses it server-side, and the
#: expression is sent on every retry), and a scan wide enough to need
#: more than this many keys is heading for the row ceiling anyway --
#: where the guard below raises with an actionable message rather than
#: letting a truncated join return a plausible wrong answer.
_PUSHDOWN_MAX_KEYS = 1024

#: SQL aggregate -> the Polars reduction that computes it. ``COUNT`` is
#: absent deliberately: ``COUNT(*)``, ``COUNT(col)`` and
#: ``COUNT(DISTINCT col)`` are three different reductions and are
#: handled explicitly in :func:`_aggregate_expr`.
_AGGREGATES: dict[type[exp.Expression], str] = {
    exp.Sum: "sum",
    exp.Avg: "mean",
    exp.Min: "min",
    exp.Max: "max",
}

#: Prefix for a projected column's positional name. Star-expanded
#: columns keep their `source.column` name instead, which is how
#: :func:`_result_names` tells the two apart.
_OUTPUT_PREFIX = "_out"

#: The hit order Milvus returned for an ANN search, carried alongside
#: the row so a join cannot lose it.
_RANK_COLUMN = "__rank"

_ARITHMETIC: dict[type[exp.Expression], str] = {
    exp.Add: "__add__",
    exp.Sub: "__sub__",
    exp.Mul: "__mul__",
    exp.Div: "__truediv__",
}


# -----------------------------------------------------------------------
# Plan shapes
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class _Search:
    """The ANN half of a scan: ``ORDER BY <column> <=> :q LIMIT k``."""

    anns_field: str
    vector: t.Any
    metric_type: str
    params: dict[str, t.Any]
    limit: int


@dataclass
class _Scan:
    """One collection read. Mutable because the planner fills
    :attr:`key_pushdown` in during a second pass, once the final scan
    order is known, and the executor fills the fetched rows in."""

    #: How the query refers to this source (its alias, or the table
    #: name) -- the prefix every column of this scan carries.
    frame: str
    #: Where this read's rows live once fetched. Distinct from
    #: :attr:`frame` because a name is only unique within one query
    #: scope: ``SELECT id FROM items UNION SELECT id FROM items`` reads
    #: ``items`` twice, with different filters, and keying the results
    #: by table name made the second read overwrite the first -- both
    #: branches then answered from the same rows.
    key: str = ""
    collection: str = ""
    fields: list[str] = field(default_factory=list)
    filter_text: str = ""
    limit: int = DEFAULT_QUERY_LIMIT
    search: _Search | None = None
    consistency: str | None = None
    star: bool = False
    #: ``(source scan key, that scan's qualified column, own column)``
    #: -- restrict this scan to the join-key values an earlier read
    #: actually returned.
    key_pushdown: tuple[str, str, str] | None = None


@dataclass(frozen=True)
class _Join:
    """One ``JOIN`` clause, already reduced to equi-key pairs plus
    whatever residual condition Polars has to evaluate itself."""

    frame: str
    how: str
    left_keys: list[str]
    right_keys: list[str]
    condition: exp.Expression | None


@dataclass(frozen=True)
class _Select:
    """A planned ``SELECT``: which frames it reads, how they combine,
    and what comes out. Nested in ``sources`` for a ``FROM (SELECT ...)``
    and in ``subqueries`` for a ``WHERE ... IN (SELECT ...)``."""

    sources: list[tuple[str, _Scan | _Select | _SetOp]]
    joins: list[_Join]
    where: exp.Expression | None
    group: list[exp.Expression]
    having: exp.Expression | None
    projections: list[exp.Expression]
    output_names: list[str]
    order: list[tuple[exp.Expression, bool]]
    limit: int | None
    offset: int | None
    distinct: bool
    #: ``<frame>.__rank`` when a vector search fed this query -- the
    #: leading sort key, so Milvus's ranking survives a join.
    search_rank: str | None = None
    #: Frames that only the predicates reference, keyed by the id of the
    #: ``exp.Subquery``/``exp.Select`` node that produced them.
    subqueries: dict[int, _Select] = field(default_factory=dict)


@dataclass(frozen=True)
class _SetOp:
    """``UNION``/``INTERSECT``/``EXCEPT``: two planned queries and the
    rule for combining their rows. Either side may itself be a set
    operation -- ``a UNION b UNION c`` nests to the left."""

    kind: str
    distinct: bool
    left: _Select | _SetOp
    right: _Select | _SetOp
    output_names: list[str]
    order: list[tuple[exp.Expression, bool]]
    limit: int | None
    offset: int | None


@dataclass(frozen=True)
class _Scope:
    """What a nested ``SELECT`` inherits from the one around it: the
    frames it could correlate against (rejected, but by name -- see
    :func:`_qualified`) and the CTEs it can read from."""

    outer: frozenset[str] = frozenset()
    ctes: dict[str, _Select | _SetOp] = field(default_factory=dict)

    def inside(self, known: set[str]) -> _Scope:
        return _Scope(self.outer | known, self.ctes)

    def adding(self, ctes: dict[str, _Select | _SetOp]) -> _Scope:
        return _Scope(self.outer, {**self.ctes, **ctes})


# -----------------------------------------------------------------------
# Should this statement take the relational path at all?
# -----------------------------------------------------------------------


def needs_relational_engine(ast: exp.Select) -> bool:
    """``True`` when the statement reads more than one collection,
    groups, or contains a subquery -- the three shapes a single
    ``query``/``search`` RPC cannot express.

    Everything else keeps going through :mod:`ast_to_pymilvus`'s
    single-call path unchanged: a plain filter ``SELECT``, a vector
    search, a hybrid search and a bare aggregate all stay exactly one
    RPC with no DataFrame anywhere near them."""
    if (
        ast.args.get("joins")
        or ast.args.get("group") is not None
        or ast.args.get("with_") is not None
        # `DISTINCT` and `OFFSET` have no Milvus-side equivalent either:
        # `query()` takes a filter and a row cap, nothing else. The
        # single-call path drops both silently, so anything carrying one
        # belongs here, where they are actually applied.
        or ast.args.get("distinct") is not None
        or ast.args.get("offset") is not None
        # `*` needs expanding against the fetched rows, which only this
        # engine does: the single-call path asked Milvus for a field
        # literally named `*` and returned one column of `None`.
        or any(
            _star_source(projection) is not _NOT_A_STAR
            for projection in ast.expressions
        )
    ):
        return True
    from_node = ast.args.get("from_")
    if from_node is not None and isinstance(from_node.this, exp.Subquery):
        # A subquery that only relabels/filters one collection is
        # flattened away before this is reached (see
        # `ast_to_pymilvus._flatten_trivial_subquery`), so anything
        # still wrapped here genuinely needs evaluating in two steps.
        return True
    # A subquery or a window function anywhere a value is expected
    # counts, not just in WHERE: Django's `annotate(Subquery(...))` puts
    # one in the SELECT list, and `ROW_NUMBER() OVER (...)` lives there
    # too. The single-call path would ask Milvus for an output field
    # named after either and hand back a column of `None`.
    scopes = [
        ast.args.get("where"),
        ast.args.get("having"),
        ast.args.get("order"),
        *ast.expressions,
    ]
    return any(
        scope is not None and scope.find(exp.Select, exp.Window) is not None
        for scope in scopes
    )


# -----------------------------------------------------------------------
# Planning
# -----------------------------------------------------------------------


def _source_name(node: exp.Expression) -> str:
    """The name the rest of the query refers to this source by: its
    alias when it has one, its table name otherwise."""
    alias = node.args.get("alias")
    if alias is not None:
        return alias.name
    if isinstance(node, exp.Table):
        return node.name
    msg = "a subquery in FROM must be aliased"
    raise errors.NotSupportedError(msg)


def _columns_of(node: exp.Expression) -> list[exp.Column]:
    """Every column reference in ``node`` belonging to *this* query.

    Deliberately does not descend into a nested ``SELECT``. Those columns
    belong to the subquery's own collections, and with a single source in
    scope :func:`_qualified` attaches any bare name to it -- so
    ``WHERE cat_id IN (SELECT id FROM categories WHERE title = :t)`` used
    to ask the *outer* collection for a ``title`` field it has never
    heard of. Milvus Lite quietly omits an unknown output field; a real
    server rejects the whole call with "field title not exist"."""
    if isinstance(node, exp.Column):
        # `t.*` is a Column too, but it names no field -- the expansion
        # happens against the fetched frame, and asking Milvus for a
        # field called `*` returns a column of nulls.
        return [] if isinstance(node.this, exp.Star) else [node]
    if isinstance(node, exp.Select):
        return []
    found: list[exp.Column] = []
    for value in node.args.values():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, exp.Expression):
                found.extend(_columns_of(item))
    return found


def _split_conjuncts(node: exp.Expression | None) -> list[exp.Expression]:
    """``a AND b AND c`` -> ``[a, b, c]``. Each conjunct can be pushed
    into a collection's own filter independently; an ``OR`` cannot be
    split this way and stays whole."""
    if node is None:
        return []
    if isinstance(node, exp.Where):
        return _split_conjuncts(node.this)
    if isinstance(node, exp.Paren):
        return _split_conjuncts(node.this)
    if isinstance(node, exp.And):
        return _split_conjuncts(node.this) + _split_conjuncts(node.expression)
    return [node]


def _referenced_frames(node: exp.Expression, known: set[str]) -> set[str]:
    """Which sources a predicate touches. An unqualified column belongs
    to the only source in scope; with several in scope it is genuinely
    ambiguous and :func:`_qualified` rejects it there."""
    frames = set()
    for column in _columns_of(node):
        table = column.table
        if table:
            frames.add(table)
        elif len(known) == 1:
            frames.add(next(iter(known)))
        else:
            frames.add("?")
    return frames


def _plan_select(
    ast: exp.Select,
    parameters: dict[str, t.Any],
    scans: list[_Scan],
    scope: _Scope | None = None,
) -> _Select:
    """Walk one ``SELECT`` into a :class:`_Select`, appending every
    collection read it needs to ``scans`` -- one list shared by the whole
    statement, so a subquery's reads join the same chain rather than
    starting a second one."""
    scope = scope or _Scope()
    scope = scope.adding(_plan_ctes(ast, parameters, scans, scope))

    join_nodes = list(ast.args.get("joins") or [])
    sources = _plan_sources(ast, join_nodes, parameters, scans, scope)
    known = {name for name, _ in sources}
    # Everything an inner SELECT could correlate against: this query's
    # sources plus whatever the enclosing ones brought.
    inner_scope = scope.inside(known)
    scan_frames = {
        name for name, source in sources if isinstance(source, _Scan)
    }

    pushable, residual, subqueries = _plan_predicates(
        ast,
        known,
        scan_frames - _null_supplying(ast, join_nodes),
        parameters,
        scans,
        inner_scope,
    )
    subqueries.update(
        _plan_value_subqueries(ast, parameters, scans, inner_scope)
    )
    order, search_frame, search = _plan_order(ast, parameters, known)

    having = ast.args.get("having")
    projections = list(ast.expressions)
    output_names = [
        projection.output_name or f"_col{index}"
        for index, projection in enumerate(projections)
    ]
    group = [
        _group_key(key, projections)
        for key in (ast.args.get("group") or exp.Group()).expressions
    ]

    # `None` in here means an unqualified `*`: every source. A `t.*`
    # contributes just that source's name, so only it is asked for
    # every field.
    star_sources = {
        _star_source(projection)
        for projection in projections
        if _star_source(projection) is not _NOT_A_STAR
    }

    # Pushed-down conjuncts are deliberately absent: Milvus applies
    # those server-side, so the columns they name need not come back.
    consumers: list[exp.Expression] = [*projections, *residual, *group]
    # An ORDER BY key that names an output label (`ORDER BY r`, for
    # `ROW_NUMBER() ... AS r`) is not a source column -- the projection
    # it refers to is already in this list, and asking Milvus for a
    # field called `r` would fail or, worse, resolve to the wrong
    # collection.
    consumers.extend(
        expression
        for expression, _ in order
        if not _names_an_output(expression, output_names)
    )
    if having is not None:
        consumers.append(having)
    consumers.extend(
        join.args["on"] for join in join_nodes if join.args.get("on")
    )
    needed = _needed_columns(consumers, known, scope.outer)

    joins = _plan_joins(ast, join_nodes, known)
    for join in joins:
        # A `USING (k)` join names no column node anywhere, and even an
        # `ON` key is only in `needed` because the condition mentions it
        # -- adding the planned keys covers both without depending on
        # how the condition was spelled.
        for key in (*join.left_keys, *join.right_keys):
            frame, column = key.split(".", 1)
            if frame in needed and column not in needed[frame]:
                needed[frame].append(column)

    consistency = ast.args.get(CONSISTENCY_ARG)
    for name, source in sources:
        if not isinstance(source, _Scan):
            continue
        source.fields = needed[name]
        source.star = None in star_sources or name in star_sources
        source.filter_text = _conjunct_filter(pushable[name], parameters)
        if search is not None and name == search_frame:
            source.search = search
            source.limit = search.limit
        if consistency is not None:
            source.consistency = consistency.this.name
        source.key = f"{source.frame}#{len(scans)}"
        scans.append(source)

    offset_node = ast.args.get("offset")
    return _Select(
        sources=sources,
        joins=joins,
        where=_conjunct_expression(residual),
        group=group,
        having=having.this if having is not None else None,
        projections=projections,
        output_names=output_names,
        order=order,
        limit=_limit_of(ast, parameters),
        offset=(
            int(resolve_value(offset_node.expression, parameters))
            if offset_node is not None
            else None
        ),
        distinct=ast.args.get("distinct") is not None,
        search_rank=(
            f"{search_frame}.{_RANK_COLUMN}" if search is not None else None
        ),
        subqueries=subqueries,
    )


def _names_an_output(node: exp.Expression, output_names: list[str]) -> bool:
    return (
        isinstance(node, exp.Column)
        and not node.table
        and node.name in output_names
    )


def _group_key(
    node: exp.Expression, projections: list[exp.Expression]
) -> exp.Expression:
    """A ``GROUP BY`` key -> the expression to group on.

    An ordinal position addresses the SELECT list, and Django's SQL
    compiler emits nothing else: every ``.values(...).annotate(...)``
    query groups by ``GROUP BY 1`` (confirmed directly against its
    compiler output), never by naming the column. Left unresolved that
    groups by the constant ``1`` -- one group for the whole
    collection -- so the projected column is then neither grouped nor
    aggregated and the query fails."""
    if isinstance(node, exp.Literal) and not node.is_string:
        index = int(node.this) - 1
        if not 0 <= index < len(projections):
            msg = f"GROUP BY position {node.this} is out of range"
            raise errors.ProgrammingError(msg)
        projection = projections[index]
        return (
            projection.this
            if isinstance(projection, exp.Alias)
            else projection
        )
    return node


def _plan_ctes(
    ast: exp.Select,
    parameters: dict[str, t.Any],
    scans: list[_Scan],
    scope: _Scope,
) -> dict[str, _Select | _SetOp]:
    """``WITH name AS (SELECT ...)`` -> a planned query per name.

    A CTE is a derived table that happens to be named up front, so it
    plans exactly like a subquery in ``FROM``: its collection is read
    once, into the same chain, and the outer query reads the result.
    Referencing one twice re-uses that single read -- the frames are
    already fetched by the time anything is evaluated."""
    with_node = ast.args.get("with_")
    if with_node is None:
        return {}
    if with_node.args.get("recursive"):
        msg = (
            "WITH RECURSIVE is not supported: a recursive query needs to "
            "re-read until a fixpoint, and Milvus is read once per plan"
        )
        raise errors.NotSupportedError(msg)
    planned: dict[str, _Select | _SetOp] = {}
    for cte in with_node.expressions:
        # Each CTE can see the ones declared before it, same as SQL.
        planned[cte.alias] = _plan_branch(
            cte.this, parameters, scans, scope.adding(planned)
        )
    return planned


def _plan_sources(
    ast: exp.Select,
    join_nodes: list[exp.Join],
    parameters: dict[str, t.Any],
    scans: list[_Scan],
    scope: _Scope,
) -> list[tuple[str, _Scan | _Select | _SetOp]]:
    """``FROM``/``JOIN`` -> one source per name. A table becomes a scan
    (or the CTE of that name, if one is in scope); an aliased subquery
    becomes a nested plan whose own scans are appended to the same
    chain."""
    from_node = ast.args.get("from_")
    if from_node is None:
        msg = "SELECT with no FROM clause is not supported"
        raise errors.NotSupportedError(msg)

    sources: list[tuple[str, _Scan | _Select | _SetOp]] = []
    for node in [from_node.this, *(join.this for join in join_nodes)]:
        name = _source_name(node)
        if isinstance(node, exp.Table) and node.name in scope.ctes:
            sources.append((name, scope.ctes[node.name]))
        elif isinstance(node, exp.Table):
            scan = _Scan(frame=name, collection=node.name, fields=[])
            sources.append((name, scan))
        elif isinstance(node, exp.Subquery):
            inner = node.this
            if not isinstance(inner, exp.Select):
                msg = (
                    "only a SELECT subquery is supported in FROM, not "
                    f"{inner.__class__.__name__}"
                )
                raise errors.NotSupportedError(msg)
            planned = _plan_select(inner, parameters, scans, scope)
            if (
                isinstance(planned, _Select)
                and _has_star(planned)
                and len(planned.sources) > 1
            ):
                # The outer query addresses this subquery's columns as
                # `alias.column`, and a starred join produces two
                # columns that share a bare name (`items.id` and
                # `cats.id` both become `id`) -- re-qualifying them
                # under one alias would collide. Naming the columns in
                # the subquery resolves it.
                msg = (
                    "SELECT * inside a subquery that joins is not "
                    "supported: name the columns it should expose"
                )
                raise errors.NotSupportedError(msg)
            sources.append((name, planned))
        else:
            msg = (
                f"unsupported FROM source: {node.__class__.__name__}; FROM "
                "must name a collection or an aliased subquery"
            )
            raise errors.NotSupportedError(msg)
    return sources


def _null_supplying(ast: exp.Select, join_nodes: list[exp.Join]) -> set[str]:
    """The sources an outer join can invent null rows for.

    A ``WHERE`` predicate on one of these must not be pushed into its
    collection's own filter: ``LEFT JOIN cats ... WHERE c.title IS NULL``
    asks for the items that matched *no* category, and Milvus filtering
    ``cats`` down to rows whose title is null answers a different
    question entirely (it did, before this existed -- every item came
    back). The predicate has to run after the join, where those nulls
    exist. Predicates on the preserved side stay pushable, and so does
    the join-key restriction in :func:`_apply_pushdown`, which only ever
    removes rows that could not have matched anything."""
    from_node = ast.args.get("from_")
    seen = [_source_name(from_node.this)] if from_node is not None else []
    null_supplying: set[str] = set()
    for join in join_nodes:
        frame = _source_name(join.this)
        side = (join.side or "").upper()
        if side == "LEFT":
            null_supplying.add(frame)
        elif side == "RIGHT":
            null_supplying.update(seen)
        elif side == "FULL":
            null_supplying.update(seen)
            null_supplying.add(frame)
        seen.append(frame)
    return null_supplying


def _plan_predicates(  # noqa: PLR0913, PLR0917 -- one argument per planning input, all needed
    ast: exp.Select,
    known: set[str],
    pushable_frames: set[str],
    parameters: dict[str, t.Any],
    scans: list[_Scan],
    scope: _Scope,
) -> tuple[
    dict[str, list[exp.Expression]],
    list[exp.Expression],
    dict[int, _Select],
]:
    """Split ``WHERE`` into what Milvus can filter and what Polars has
    to evaluate.

    A conjunct that mentions exactly one collection and nothing else
    becomes part of that collection's own ``filter``; one that spans two
    collections, contains a subquery, or names a source an outer join
    can null-extend (see :func:`_null_supplying`) stays behind as
    residual."""
    subqueries: dict[int, _Select] = {}
    residual: list[exp.Expression] = []
    pushable: dict[str, list[exp.Expression]] = {name: [] for name in known}
    for conjunct in _split_conjuncts(ast.args.get("where")):
        nested = _outermost_selects(conjunct)
        if nested:
            for inner in nested:
                subqueries[id(inner)] = _plan_select(
                    inner, parameters, scans, scope
                )
            residual.append(conjunct)
            continue
        frames = _referenced_frames(conjunct, known)
        if len(frames) == 1 and frames <= pushable_frames:
            pushable[next(iter(frames))].append(conjunct)
        else:
            residual.append(conjunct)
    return pushable, residual, subqueries


def _plan_value_subqueries(
    ast: exp.Select,
    parameters: dict[str, t.Any],
    scans: list[_Scan],
    scope: _Scope,
) -> dict[int, _Select]:
    """Subqueries outside ``WHERE`` -- in the SELECT list, ``HAVING`` or
    ``ORDER BY`` -- each planned as one more read in the same chain."""
    planned: dict[int, _Select] = {}
    hosts: list[exp.Expression] = [*ast.expressions]
    for key in ("having", "order"):
        node = ast.args.get(key)
        if node is not None:
            hosts.append(node)
    for host in hosts:
        for inner in _outermost_selects(host):
            planned[id(inner)] = _plan_select(inner, parameters, scans, scope)
    return planned


def _outermost_selects(node: exp.Expression) -> list[exp.Select]:
    """Every ``SELECT`` directly inside ``node``, without descending
    into one once found -- a subquery nested inside another subquery is
    that subquery's business, and planning it twice here would fetch its
    collection twice."""
    if isinstance(node, exp.Select):
        return [node]
    found: list[exp.Select] = []
    for value in node.args.values():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, exp.Expression):
                found.extend(_outermost_selects(item))
    return found


def _plan_order(
    ast: exp.Select, parameters: dict[str, t.Any], known: set[str]
) -> tuple[list[tuple[exp.Expression, bool]], str | None, _Search | None]:
    """``ORDER BY`` -> sort keys, plus the ANN search when the leading
    key is a vector distance rather than a scalar."""
    order_node = ast.args.get("order")
    if order_node is None:
        return [], None, None
    keys = order_node.expressions
    metric_type = METRIC_TYPES.get(type(keys[0].this))
    if metric_type is None:
        return (
            [(key.this, bool(key.args.get("desc"))) for key in keys],
            None,
            None,
        )
    search_frame, search = _plan_search(
        keys[0].this, metric_type, ast, parameters, known
    )
    return (
        [(key.this, bool(key.args.get("desc"))) for key in keys[1:]],
        search_frame,
        search,
    )


def _needed_columns(
    consumers: list[exp.Expression],
    known: set[str],
    outer: frozenset[str],
) -> dict[str, list[str]]:
    """Projection pushdown: the fields each collection is asked for are
    exactly the ones the statement mentions."""
    needed: dict[str, list[str]] = {name: [] for name in known}
    for consumer in consumers:
        for column in _columns_of(consumer):
            frame, name = _qualified(column, known, outer)
            if name not in needed[frame]:
                needed[frame].append(name)
    return needed


def _limit_of(ast: exp.Select, parameters: dict[str, t.Any]) -> int | None:
    """``LIMIT`` as a number. It is a literal in hand-written MilvusQL
    but a bound parameter in SQLAlchemy-generated text (confirmed
    directly: ``.limit(5)`` compiles to ``LIMIT :param_1``), so it
    resolves through the same value path as everything else."""
    limit_node = ast.args.get("limit")
    if limit_node is None:
        return None
    return int(resolve_value(limit_node.expression, parameters))


def _plan_search(
    distance: exp.Expression,
    metric_type: str,
    ast: exp.Select,
    parameters: dict[str, t.Any],
    known: set[str],
) -> tuple[str, _Search]:
    """``ORDER BY t.embedding <=> :q LIMIT k`` -> the ANN search on
    ``t``'s scan.

    In a joined query this makes the ``LIMIT`` mean "Milvus's top-k from
    *that* collection, then join", not "top-k of the joined result" --
    the two only differ when the join or a cross-collection predicate
    drops rows, and the alternative (rank after joining) would mean
    fetching the whole vector collection, which is the one thing an ANN
    index exists to avoid. Documented in the README rather than guessed
    at silently."""
    limit = _limit_of(ast, parameters)
    if limit is None:
        msg = (
            "a vector search needs a LIMIT: ORDER BY <vector distance> "
            "asks Milvus for the top matches, and how many is not optional"
        )
        raise errors.NotSupportedError(msg)
    column = distance.this
    if not isinstance(column, exp.Column):
        msg = "the left side of a vector distance must be a column"
        raise errors.NotSupportedError(msg)
    frame, name = _qualified(column, known)
    params_node = ast.args.get(SEARCH_PARAMS_ARG)
    knobs = {
        prop.this.name: resolve_value(prop.args["value"], parameters)
        for prop in (params_node.expressions if params_node else [])
    }
    return frame, _Search(
        anns_field=name,
        vector=resolve_value(distance.expression, parameters),
        metric_type=metric_type,
        params=knobs,
        limit=limit,
    )


def _plan_joins(
    ast: exp.Select, join_nodes: list[exp.Join], known: set[str]
) -> list[_Join]:
    """Plan each ``JOIN`` in order, telling it which sources precede it
    -- ``USING`` needs that to know which frame owns the left key."""
    from_node = ast.args.get("from_")
    preceding = [_source_name(from_node.this)] if from_node else []
    planned = []
    for join in join_nodes:
        planned.append(_plan_join(join, known, preceding))
        preceding.append(_source_name(join.this))
    return planned


def _plan_join(join: exp.Join, known: set[str], preceding: list[str]) -> _Join:
    frame = _source_name(join.this)
    side = (join.side or "").upper()
    kind = (join.kind or "").upper()
    on = join.args.get("on")
    how = {"LEFT": "left", "RIGHT": "right", "FULL": "full"}.get(side, "inner")
    using = join.args.get("using")
    if using:
        # `USING (k)` is `ON left.k = right.k` with the left side left
        # implicit. With one source before this join there is only one
        # frame it can mean; with several there is no schema here to say
        # which of them owns `k`, and guessing would join on the wrong
        # collection's column.
        if len(preceding) != 1:
            msg = (
                "JOIN ... USING is supported for two sources only -- with "
                f"{len(preceding)} already joined there is no schema here "
                "to say which one owns the key. Spell it as ON"
            )
            raise errors.NotSupportedError(msg)
        names = [identifier.name for identifier in using]
        return _Join(
            frame,
            how,
            [f"{preceding[0]}.{name}" for name in names],
            [f"{frame}.{name}" for name in names],
            None,
        )
    if kind == "CROSS" or on is None:
        return _Join(frame, "cross", [], [], None)

    left_keys: list[str] = []
    right_keys: list[str] = []
    residual: list[exp.Expression] = []
    for conjunct in _split_conjuncts(on):
        pair = _equi_key_pair(conjunct, frame, known)
        if pair is None:
            residual.append(conjunct)
            continue
        left, right = pair
        left_keys.append(left)
        right_keys.append(right)
    if not left_keys:
        # No equality to hash on -- the pairing is every combination,
        # narrowed by the condition, which is what a SQL engine falls
        # back to as well. `how` is kept rather than degraded to a cross
        # join: an outer join still has to preserve its side, which
        # `_join` handles. Small inputs only; the row ceiling keeps that
        # honest.
        return _Join(
            frame, how, [], [], _conjunct_expression(_split_conjuncts(on))
        )
    return _Join(
        frame, how, left_keys, right_keys, _conjunct_expression(residual)
    )


def _equi_key_pair(
    conjunct: exp.Expression, frame: str, known: set[str]
) -> tuple[str, str] | None:
    """``a.x = b.y`` -> the two qualified column names, oriented so the
    right-hand one belongs to the frame being joined in."""
    if not isinstance(conjunct, exp.EQ):
        return None
    left, right = conjunct.this, conjunct.expression
    if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
        return None
    left_frame, left_name = _qualified(left, known)
    right_frame, right_name = _qualified(right, known)
    if left_frame == frame and right_frame != frame:
        return f"{right_frame}.{right_name}", f"{left_frame}.{left_name}"
    if right_frame == frame and left_frame != frame:
        return f"{left_frame}.{left_name}", f"{right_frame}.{right_name}"
    return None


def _qualified(
    column: exp.Column, known: set[str], outer: frozenset[str] = frozenset()
) -> tuple[str, str]:
    """``t.col`` -> ``("t", "col")``, and a bare ``col`` -> the only
    source in scope.

    With several sources in scope an unqualified column cannot be
    resolved here: Milvus is asked for fields *before* any row comes
    back, and no collection schema is available at translate time to say
    which one owns the name. Guessing would silently attach the column
    to the wrong collection, so this asks for the qualification
    instead."""
    if column.table:
        if column.table in outer and column.table not in known:
            # A correlated subquery: it names a table from the query
            # that encloses it, so its result depends on the row being
            # evaluated. Answering that needs either one Milvus read per
            # outer row or a rewrite into a semi-join -- neither exists
            # here yet, and the ORMs that emit this shape
            # (SQLAlchemy's `.any()`/`.has()`, Django's `Exists()` with
            # `OuterRef`) deserve to be told which construct is missing.
            msg = (
                "correlated subqueries are not supported: this one "
                f"references {column.table!r} from the enclosing query. "
                "Rewrite it as a JOIN, or as an uncorrelated "
                "IN (SELECT ...)"
            )
            raise errors.NotSupportedError(msg)
        if column.table not in known:
            msg = (
                f"unknown table qualifier {column.table!r} on column "
                f"{column.name!r}"
            )
            raise errors.ProgrammingError(msg)
        return column.table, column.name
    if len(known) == 1:
        return next(iter(known)), column.name
    msg = (
        f"ambiguous column {column.name!r}: qualify it with its table "
        f"(one of {', '.join(sorted(known))}) -- with more than one "
        "collection in scope there is no schema at translate time to "
        "resolve a bare column name against"
    )
    raise errors.ProgrammingError(msg)


def _conjunct_expression(
    conjuncts: list[exp.Expression],
) -> exp.Expression | None:
    if not conjuncts:
        return None
    combined = conjuncts[0]
    for conjunct in conjuncts[1:]:
        combined = exp.and_(combined, conjunct)
    return combined


def _conjunct_filter(
    conjuncts: list[exp.Expression], parameters: dict[str, t.Any]
) -> str:
    """Render pushed-down conjuncts into one Milvus filter string. The
    column qualifier (``t.price``) is dropped on the way -- Milvus knows
    its own fields by bare name."""
    if not conjuncts:
        return ""
    stripped = []
    for conjunct in conjuncts:
        copied = conjunct.copy()
        for column in copied.find_all(exp.Column):
            column.set("table", None)
        stripped.append(copied)
    combined = stripped[0]
    for conjunct in stripped[1:]:
        combined = exp.and_(combined, conjunct)
    return render_filter(combined, parameters)


def _nested_plans(plan: _Select | _SetOp) -> list[_Select | _SetOp]:
    """Every planned query one level inside this one: derived tables,
    CTEs used as sources, predicate subqueries, set-operation
    branches."""
    if isinstance(plan, _SetOp):
        return [plan.left, plan.right]
    nested: list[_Select | _SetOp] = [
        source
        for _name, source in plan.sources
        if not isinstance(source, _Scan)
    ]
    nested.extend(plan.subqueries.values())
    return nested


def _order_scans(scans: list[_Scan]) -> list[_Scan]:
    """Run the most selective scans first: an ANN search (bounded by its
    own top-k) before a filtered scan, a filtered scan before an
    unfiltered one. Everything after the first scan can then have join
    keys pushed into it."""

    def rank(scan: _Scan) -> int:
        if scan.search is not None:
            return 0
        return 1 if scan.filter_text else 2

    return sorted(scans, key=rank)


def _plan_pushdown(plan: _Select | _SetOp, scans: list[_Scan]) -> None:
    """Fill in :attr:`_Scan.key_pushdown` wherever a scan's equi-join
    key is already known from a scan that runs before it.

    Only safe towards the side of the join whose unmatched rows are
    dropped anyway: both sides of an ``INNER`` join, the right side of a
    ``LEFT`` join, the left side of a ``RIGHT`` join. A ``FULL`` join
    keeps unmatched rows from both sides, so restricting either one
    would delete rows the query asked for."""
    for nested in _nested_plans(plan):
        _plan_pushdown(nested, scans)
    if not isinstance(plan, _Select):
        return
    position = {scan.key: index for index, scan in enumerate(scans)}
    # Only this query's own sources: a name means nothing outside the
    # scope that declared it.
    by_frame = {
        name: source
        for name, source in plan.sources
        if isinstance(source, _Scan)
    }

    for join in plan.joins:
        for left, right in zip(join.left_keys, join.right_keys, strict=True):
            left_frame, left_column = left.split(".", 1)
            right_frame, right_column = right.split(".", 1)
            candidates = []
            if join.how in {"inner", "left"}:
                candidates.append(
                    (left_frame, left_column, right_frame, right_column)
                )
            if join.how in {"inner", "right"}:
                candidates.append(
                    (right_frame, right_column, left_frame, left_column)
                )
            for src_frame, src_col, dst_frame, dst_col in candidates:
                source = by_frame.get(src_frame)
                target = by_frame.get(dst_frame)
                if source is None or target is None:
                    continue
                source_at = position.get(source.key)
                target_at = position.get(target.key)
                if (
                    source_at is None
                    or target_at is None
                    or source_at >= target_at
                    or target.search is not None
                    or target.key_pushdown is not None
                ):
                    continue
                target.key_pushdown = (
                    source.key,
                    f"{source.frame}.{src_col}",
                    dst_col,
                )


# -----------------------------------------------------------------------
# Execution: one Call per scan, then evaluate what was fetched
# -----------------------------------------------------------------------


def build_relational_call(
    ast: exp.Select, parameters: dict[str, t.Any]
) -> Call:
    """A multi-collection / grouped / subquery ``SELECT`` -> the chain of
    ``Call``s that answers it."""
    scans: list[_Scan] = []
    plan = _plan_select(ast, parameters, scans)
    scans = _order_scans(scans)
    _plan_pushdown(plan, scans)
    return _scan_call(0, scans, {}, plan)


def build_set_operation_call(
    ast: exp.SetOperation, parameters: dict[str, t.Any]
) -> Call:
    """``SELECT ... UNION SELECT ...`` -> the chain that answers it.

    Each branch is planned like any other query -- its reads join the
    same chain -- and the combination happens once both frames are in
    hand."""
    scans: list[_Scan] = []
    plan = _plan_set_operation(ast, parameters, scans, _Scope())
    scans = _order_scans(scans)
    # A branch can join, and its joins deserve the same key pushdown
    # every other query gets.
    _plan_pushdown(plan, scans)
    return _scan_call(0, scans, {}, plan)


_SET_OPERATIONS: dict[type[exp.Expression], str] = {
    exp.Union: "union",
    exp.Intersect: "intersect",
    exp.Except: "except",
}


def _plan_set_operation(
    ast: exp.SetOperation,
    parameters: dict[str, t.Any],
    scans: list[_Scan],
    scope: _Scope,
) -> _SetOp:
    kind = _SET_OPERATIONS[type(ast)]
    distinct = bool(ast.args.get("distinct"))
    if kind != "union" and not distinct:
        # `INTERSECT ALL`/`EXCEPT ALL` are multiset operations -- they
        # keep min/difference *counts* of duplicate rows, which a
        # semi/anti join cannot express. Saying so beats quietly
        # answering the DISTINCT version.
        msg = (
            f"{kind.upper()} ALL is not supported: its duplicate-count "
            f"semantics need a multiset operation, not a filter. Use "
            f"{kind.upper()} (distinct)"
        )
        raise errors.NotSupportedError(msg)
    left = _plan_branch(ast.this, parameters, scans, scope)
    right = _plan_branch(ast.expression, parameters, scans, scope)
    order_node = ast.args.get("order")
    offset_node = ast.args.get("offset")
    return _SetOp(
        kind=kind,
        distinct=distinct,
        left=left,
        right=right,
        # SQL takes the result's column names from the first branch.
        output_names=left.output_names,
        order=[
            (key.this, bool(key.args.get("desc")))
            for key in (order_node.expressions if order_node else [])
        ],
        limit=_limit_of(ast, parameters),
        offset=(
            int(resolve_value(offset_node.expression, parameters))
            if offset_node is not None
            else None
        ),
    )


def _plan_branch(
    node: exp.Expression,
    parameters: dict[str, t.Any],
    scans: list[_Scan],
    scope: _Scope,
) -> _Select | _SetOp:
    if isinstance(node, exp.SetOperation):
        return _plan_set_operation(node, parameters, scans, scope)
    if isinstance(node, exp.Subquery):
        return _plan_branch(node.this, parameters, scans, scope)
    if isinstance(node, exp.Select):
        return _plan_select(node, parameters, scans, scope)
    msg = f"unsupported set-operation branch: {node.__class__.__name__}"
    raise errors.NotSupportedError(msg)


def _scan_call(
    index: int,
    scans: list[_Scan],
    frames: dict[str, pl.DataFrame],
    plan: _Select | _SetOp,
) -> Call:
    """The ``Call`` for ``scans[index]``, chained to the next one.

    ``frames`` is the accumulator the chain writes into: each link
    records its own rows and then either builds the next link -- late
    enough for :func:`_apply_pushdown` to use the rows just fetched --
    or, for the last link, evaluates the plan in ``postprocess``."""
    scan = scans[index]
    _apply_pushdown(scan, frames)
    method, kwargs = _call_args(scan)

    if index == len(scans) - 1:

        def postprocess(raw: t.Any) -> RowsAndDescription:  # noqa: ANN401
            frames[scan.key] = _frame(scan, raw)
            return _finish(plan, frames)

        return Call(method, kwargs, postprocess)

    def then(raw: t.Any) -> Call:  # noqa: ANN401
        frames[scan.key] = _frame(scan, raw)
        return _scan_call(index + 1, scans, frames, plan)

    return Call(method, kwargs, then=then)


def _call_args(scan: _Scan) -> tuple[str, dict[str, t.Any]]:
    kwargs: dict[str, t.Any] = {"collection_name": scan.collection}
    if scan.consistency is not None:
        kwargs["consistency_level"] = scan.consistency
    # No named column at all means the query only counts this side's
    # rows (`SELECT COUNT(*) FROM a CROSS JOIN b`). Milvus has no
    # "fetch nothing but the row count of a filtered scan" output, and
    # no schema is available here to pick a cheap field, so `*` is the
    # honest request -- correct, and paid for only by a query that
    # references none of the collection's columns.
    output_fields = (
        ["*"] if scan.star or not scan.fields else list(scan.fields)
    )
    if scan.search is not None:
        kwargs.update(
            data=[scan.search.vector],
            anns_field=scan.search.anns_field,
            limit=scan.search.limit,
            output_fields=output_fields,
            search_params={
                "metric_type": scan.search.metric_type,
                "params": scan.search.params,
            },
        )
        if scan.filter_text:
            kwargs["filter"] = scan.filter_text
        return "search", kwargs
    kwargs["output_fields"] = output_fields
    kwargs["limit"] = scan.limit
    if scan.filter_text:
        kwargs["filter"] = scan.filter_text
    return "query", kwargs


def _apply_pushdown(scan: _Scan, frames: dict[str, pl.DataFrame]) -> None:
    """Narrow a scan's filter to the join-key values an earlier scan
    actually returned. Skipped past :data:`_PUSHDOWN_MAX_KEYS` keys: the
    query stays correct either way, it just reads more rows and may meet
    the ceiling guard instead."""
    if scan.key_pushdown is None:
        return
    source_key, column, own_column = scan.key_pushdown
    frame = frames.get(source_key)
    if frame is None or column not in frame.columns:
        return
    keys = frame[column].drop_nulls().unique().to_list()
    if len(keys) > _PUSHDOWN_MAX_KEYS:
        return
    if keys:
        values = ", ".join(render_filter_value(key) for key in keys)
        rendered = f"{own_column} in [{values}]"
    else:
        # The driving side matched nothing, so no row of this side can
        # survive the join -- say so in the filter rather than reading
        # the collection to throw all of it away.
        rendered = "false"
    scan.filter_text = (
        f"({scan.filter_text}) and ({rendered})"
        if scan.filter_text
        else rendered
    )


def _frame(scan: _Scan, raw: t.Any) -> pl.DataFrame:  # noqa: ANN401
    """One scan's raw pymilvus result -> a frame whose columns carry the
    source's name (``items.id``), so every later expression resolves
    against one flat namespace and no join has to invent suffixes."""
    if scan.search is not None:
        hits = raw[0] if raw else []
        rows = [
            {
                **hit.get("entity", {}),
                "id": hit.get("id"),
                "distance": hit.get("distance"),
                # Milvus returned these already ranked. A join reorders
                # rows freely, so the position is carried as a column
                # rather than trusted to survive -- `_sort` restores it
                # afterwards. Not a metric comparison: whether nearer
                # means a larger or a smaller `distance` depends on the
                # index's metric type, but the order Milvus sent is the
                # order it means.
                f"{_RANK_COLUMN}": rank,
            }
            for rank, hit in enumerate(hits)
        ]
    else:
        rows = list(raw or [])
        if len(rows) >= scan.limit:
            msg = (
                f"cannot be answered: reading {scan.collection!r} hit "
                f"Milvus's per-call row ceiling ({scan.limit}), so the "
                "join/grouping would run over a truncated subset and "
                "return a wrong answer with nothing to show it is wrong. "
                "Narrow the WHERE filter, or add a vector search so "
                "Milvus returns a bounded top-k."
            )
            raise errors.NotSupportedError(msg)

    columns = list(scan.fields)
    if scan.search is not None:
        columns = [*columns, _RANK_COLUMN]
    if scan.star or not columns:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen or columns
    data = {
        f"{scan.frame}.{column}": [row.get(column) for row in rows]
        for column in columns
    }
    # An empty result still needs its columns present: a join against it
    # has to produce nulls, not a missing-column error. The dtype is
    # unknowable with no values to infer from, which is what `_align`
    # below repairs at join time.
    return pl.DataFrame(data if rows else {name: [] for name in data})


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class _Ctx:
    """What an expression translator needs: the plan it belongs to (for
    subqueries and output names), every fetched frame (for subquery
    evaluation), the frame being translated against, and -- after a
    ``GROUP BY`` -- the substitution that maps grouped/aggregated
    sub-expressions onto the columns the aggregation produced."""

    plan: _Select
    frames: dict[str, pl.DataFrame]
    frame: pl.DataFrame
    substitute: t.Callable[[exp.Expression], pl.Expr | None] | None = None

    def against(self, frame: pl.DataFrame) -> _Ctx:
        return _Ctx(self.plan, self.frames, frame, self.substitute)

    def with_substitution(
        self, substitute: t.Callable[[exp.Expression], pl.Expr | None]
    ) -> _Ctx:
        return _Ctx(self.plan, self.frames, self.frame, substitute)


def _finish(
    plan: _Select | _SetOp, frames: dict[str, pl.DataFrame]
) -> RowsAndDescription:
    frame = _evaluate(plan, frames)
    rows = [tuple(row) for row in frame.rows()]
    # `SELECT *` learns its column names from the data, so the
    # description comes off the evaluated frame rather than the plan.
    return rows, description(_result_names(plan, frame)), len(rows), None


def _result_names(plan: _Select | _SetOp, frame: pl.DataFrame) -> list[str]:
    """What ``cursor.description`` calls each column.

    A projected column carries its position (``_out3``) and looks its
    label up in the plan; a star-expanded one carries its source
    qualifier (``items.id``) and drops it here, where SQL is happy to
    repeat a name that a DataFrame cannot."""
    labels = _leftmost(plan).output_names
    names = []
    for column in frame.columns:
        if column.startswith(_OUTPUT_PREFIX):
            index = int(column[len(_OUTPUT_PREFIX) :])
            names.append(labels[index] if index < len(labels) else column)
        else:
            names.append(column.split(".", 1)[-1])
    return names


def _leftmost(plan: _Select | _SetOp) -> _Select:
    """A set operation takes its column names from its first branch,
    which may itself be one."""
    while isinstance(plan, _SetOp):
        plan = plan.left
    return plan


def _has_star(plan: _Select) -> bool:
    return any(_star_source(p) is not _NOT_A_STAR for p in plan.projections)


#: Distinguishes "not a star" from "an unqualified star", which is
#: itself a meaningful value (every source).
_NOT_A_STAR = object()


def _star_source(node: exp.Expression) -> t.Any:  # noqa: ANN401
    """``*`` -> ``None`` (every source), ``t.*`` -> ``"t"``, anything
    else -> :data:`_NOT_A_STAR`. A qualified star parses as a ``Column``
    whose own ``this`` is the ``Star``, not as a ``Star`` -- missing
    that made ``SELECT i.*`` ask Milvus for a field named ``*`` and hand
    back a column of ``None``."""
    if isinstance(node, exp.Star):
        return None
    if isinstance(node, exp.Column) and isinstance(node.this, exp.Star):
        return node.table or None
    return _NOT_A_STAR


def _evaluate(
    plan: _Select | _SetOp, frames: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    """The relational algebra, in Polars, over the already-fetched
    frames -- in SQL's own clause order: FROM/JOIN, WHERE, GROUP BY,
    HAVING, ORDER BY, SELECT, DISTINCT, OFFSET/LIMIT.

    ``ORDER BY`` runs *before* the projection, not after: SQL lets it
    name a column the SELECT list never returns (``SELECT title ...
    ORDER BY price``), and once the frame has been narrowed to the
    projected columns that key is gone. Grouped queries sort inside
    :func:`_aggregate` for the same reason, one step later -- there the
    the sortable columns are the groups and the reductions."""
    if isinstance(plan, _SetOp):
        return _evaluate_set_operation(plan, frames)
    name, first = plan.sources[0]
    frame = _source_frame(name, first, frames)
    by_name = dict(plan.sources)
    for join in plan.joins:
        right = _source_frame(join.frame, by_name[join.frame], frames)
        frame = _join(frame, right, join, plan, frames)

    ctx = _Ctx(plan, frames, frame)
    if plan.where is not None:
        frame = frame.filter(_expression(plan.where, ctx))
        ctx = ctx.against(frame)

    if plan.group or _has_aggregate(plan):
        frame = _aggregate(frame, ctx)
    else:
        if plan.order or plan.search_rank:
            frame = _sort(frame, ctx.against(frame))
            ctx = ctx.against(frame)
        frame = _project(frame, ctx)

    if plan.distinct:
        frame = frame.unique(maintain_order=True)
    if plan.offset:
        frame = frame.slice(plan.offset)
    if plan.limit is not None:
        frame = frame.head(plan.limit)
    return frame


def _evaluate_set_operation(
    plan: _SetOp, frames: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    """Combine two branches. Both come out of :func:`_project` with the
    same positional column names, so they line up by position the way
    SQL says they should -- no name matching involved."""
    left = _evaluate(plan.left, frames)
    right = _evaluate(plan.right, frames)
    if left.width != right.width:
        msg = (
            f"{plan.kind.upper()} needs both sides to select the same "
            f"number of columns, got {left.width} and {right.width}"
        )
        raise errors.ProgrammingError(msg)
    right = right.rename(dict(zip(right.columns, left.columns, strict=True)))

    if plan.kind == "union":
        # `vertical_relaxed` so a branch that came back empty (dtype
        # `Null`, nothing to infer from) still concatenates.
        combined = pl.concat([left, right], how="vertical_relaxed")
    else:
        how = "semi" if plan.kind == "intersect" else "anti"
        combined = left.join(
            right,
            on=left.columns,
            how=how,
            # SQL's INTERSECT/EXCEPT compare rows with NULL = NULL,
            # unlike a join's ON -- two null-valued rows are the same
            # row here.
            nulls_equal=True,
        )
    if plan.distinct:
        combined = combined.unique(maintain_order=True)

    if plan.order:
        combined = _sort_positional(combined, plan)
    if plan.offset:
        combined = combined.slice(plan.offset)
    if plan.limit is not None:
        combined = combined.head(plan.limit)
    return combined


def _sort_positional(frame: pl.DataFrame, plan: _SetOp) -> pl.DataFrame:
    """``ORDER BY`` on a set operation can only address the result's own
    columns -- by position (``ORDER BY 1``) or by the name the first
    branch gave them. Neither branch's source columns are reachable
    here, which is exactly what SQL says too."""
    by: list[str] = []
    descending: list[bool] = []
    for node, desc in plan.order:
        if isinstance(node, exp.Literal) and not node.is_string:
            index = int(node.this) - 1
        elif isinstance(node, exp.Column) and node.name in plan.output_names:
            index = plan.output_names.index(node.name)
        else:
            msg = (
                "ORDER BY on a set operation must name a result column "
                f"or its position, not {node.sql()!r}"
            )
            raise errors.ProgrammingError(msg)
        if not 0 <= index < frame.width:
            msg = f"ORDER BY position {index + 1} is out of range"
            raise errors.ProgrammingError(msg)
        by.append(frame.columns[index])
        descending.append(desc)
    return frame.sort(
        by,
        descending=descending,
        nulls_last=[not desc for desc in descending],
        maintain_order=True,
    )


def _project(frame: pl.DataFrame, ctx: _Ctx) -> pl.DataFrame:
    """The SELECT list. Output columns are named positionally
    (``_out0``, ``_out1``, ...) rather than by their SQL labels: rows go
    back to the caller as tuples with ``cursor.description`` carrying the
    names, and SQL happily labels two columns the same (``SELECT i.id,
    c.id``) where a DataFrame cannot."""
    plan = ctx.plan
    selected: list[pl.Expr] = []
    for index, projection in enumerate(plan.projections):
        source = _star_source(projection)
        if source is _NOT_A_STAR:
            selected.append(
                _expression(projection, ctx).alias(f"{_OUTPUT_PREFIX}{index}")
            )
            continue
        # A star expands to the columns of the frame it names -- every
        # source for `*`, one for `t.*`. They keep their qualifier: two
        # collections can both have an `id`, and a DataFrame cannot hold
        # that name twice. `_result_names` strips it for the caller,
        # where SQL does allow the repeat.
        prefix = None if source is None else f"{source}."
        expanded = [
            column
            for column in frame.columns
            if not column.endswith(f".{_RANK_COLUMN}")
            and (prefix is None or column.startswith(prefix))
        ]
        if not expanded:
            msg = f"unknown table qualifier {source!r} in {source!r}.*"
            raise errors.ProgrammingError(msg)
        selected.extend(pl.col(column) for column in expanded)
    return frame.select(selected)


def _sort(frame: pl.DataFrame, ctx: _Ctx) -> pl.DataFrame:
    by: list[pl.Expr] = []
    descending: list[bool] = []
    rank = getattr(ctx.plan, "search_rank", None)
    if rank is not None and rank in frame.columns:
        # `ORDER BY <vector> <=> :q` is the first key it was written as,
        # so it stays the first key here; anything after it breaks ties.
        by.append(pl.col(rank))
        descending.append(False)
    for expression, desc in ctx.plan.order:
        by.append(_expression(_order_node(expression, ctx), ctx))
        descending.append(desc)
    return frame.sort(
        by,
        descending=descending,
        # Matches the single-collection ORDER BY path in
        # `ast_to_pymilvus._sorted_query_rows`, which sorts on
        # `(value is None, value)`: nulls last ascending, first
        # descending.
        nulls_last=[not desc for desc in descending],
        maintain_order=True,
    )


def _order_node(node: exp.Expression, ctx: _Ctx) -> exp.Expression:
    """An ``ORDER BY`` key -> the expression to actually sort on.

    Two spellings address the SELECT list rather than the source: an
    ordinal position (``ORDER BY 1``, which Django emits alongside
    ``.values()``) and an output label (``ORDER BY n`` for
    ``COUNT(*) AS n``). Both resolve back to the projection they name,
    which is what makes them sortable before the projection has been
    applied."""
    plan = ctx.plan
    if isinstance(node, exp.Literal) and not node.is_string:
        index = int(node.this) - 1
        if not 0 <= index < len(plan.projections):
            msg = f"ORDER BY position {node.this} is out of range"
            raise errors.ProgrammingError(msg)
        return plan.projections[index]
    if isinstance(node, exp.Column) and not node.table:
        for projection, name in zip(
            plan.projections, plan.output_names, strict=True
        ):
            # A label only wins when it is not also a real column of the
            # frame being sorted -- `ORDER BY price` means the column,
            # even if something else was labelled `price`.
            if name == node.name and not _resolvable(node, ctx):
                return projection
    return node


def _resolvable(node: exp.Column, ctx: _Ctx) -> bool:
    try:
        _frame_column(node, ctx)
    except errors.ProgrammingError:
        return False
    return True


def _source_frame(
    name: str,
    source: _Scan | _Select | _SetOp,
    frames: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    if isinstance(source, _Scan):
        return frames[source.key]
    inner = _evaluate(source, frames)
    # A derived table is addressed as `alias.column`, where `column` is
    # the *label* the inner SELECT gave it -- `_project` names its
    # output positionally (see there for why), so the labels are put
    # back on here, in order, before the alias is applied.
    # A starred inner query names its columns from the data, so its
    # labels come off the frame; anything else uses the labels it
    # planned. Either way they are re-qualified under this alias --
    # leaving a starred subquery's columns bare made the outer query's
    # `s.cid` a `KeyError` out of the join.
    bare = (
        [column.split(".", 1)[-1] for column in inner.columns]
        if len(source.output_names) != inner.width
        or (isinstance(source, _Select) and _has_star(source))
        else source.output_names
    )
    labels = [f"{name}.{label}" for label in bare]
    return inner.rename(dict(zip(inner.columns, labels, strict=True)))


def _join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    join: _Join,
    plan: _Select,
    frames: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    if join.condition is None:
        return _pair(left, right, join, how=join.how)
    if join.how in {"inner", "cross"}:
        # Nothing is null-extended by an inner join, so narrowing the
        # paired rows afterwards is the same thing SQL means.
        joined = _pair(left, right, join, how="inner")
        return joined.filter(
            _expression(join.condition, _Ctx(plan, frames, joined))
        )
    return _outer_join_with_condition(left, right, join, plan, frames)


def _pair(
    left: pl.DataFrame, right: pl.DataFrame, join: _Join, how: str
) -> pl.DataFrame:
    if how == "cross" or not join.left_keys:
        # No equality to hash on: the pairing is every combination, and
        # whatever condition came with it narrows them afterwards.
        return left.join(right, how="cross")
    left, right = _align(left, right, join.left_keys, join.right_keys)
    return left.join(
        right,
        left_on=join.left_keys,
        right_on=join.right_keys,
        how=t.cast("t.Any", how),
        # SQL's equi-join never matches NULL to NULL. Polars agrees
        # by default; spelled out because it is load-bearing.
        nulls_equal=False,
        # Both sides' keys are already uniquely named (`a.k` vs
        # `b.k`), so there is nothing to coalesce and both stay
        # addressable afterwards.
        coalesce=False,
        maintain_order="left",
    )


#: Row identity, added just long enough to work out which rows of an
#: outer join's preserved side found no partner.
_LEFT_TAG = "__left_row"
_RIGHT_TAG = "__right_row"


def _outer_join_with_condition(
    left: pl.DataFrame,
    right: pl.DataFrame,
    join: _Join,
    plan: _Select,
    frames: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """``LEFT``/``RIGHT``/``FULL JOIN ... ON <equality> AND <more>``, or
    an outer join whose ``ON`` is not an equality at all.

    Such a condition cannot be a hash-join key, so it has to be
    evaluated over the paired rows. Doing that as a plain filter is
    right for an inner join and wrong for this one: it deletes the
    null-extended rows the outer join exists to keep, quietly turning
    `LEFT JOIN ... ON a.k = b.k AND b.active` into an inner join. So the
    rows are paired and narrowed first, then every unpaired row of the
    preserved side is put back, null-extended."""
    condition = join.condition
    if condition is None:  # pragma: no cover -- guarded by the caller
        msg = "an outer join with no condition needs no restoration"
        raise errors.InternalError(msg)
    tagged_left = left.with_row_index(_LEFT_TAG)
    tagged_right = right.with_row_index(_RIGHT_TAG)
    matched = _pair(tagged_left, tagged_right, join, how="inner")
    matched = matched.filter(
        _expression(condition, _Ctx(plan, frames, matched))
    )
    columns = matched.columns

    parts = [matched]
    if join.how in {"left", "full"}:
        parts.append(_unmatched(tagged_left, matched, _LEFT_TAG, columns))
    if join.how in {"right", "full"}:
        parts.append(_unmatched(tagged_right, matched, _RIGHT_TAG, columns))
    combined = pl.concat(parts, how="vertical_relaxed")
    return combined.drop(_LEFT_TAG, _RIGHT_TAG)


def _unmatched(
    preserved: pl.DataFrame,
    matched: pl.DataFrame,
    tag: str,
    columns: list[str],
) -> pl.DataFrame:
    """The preserved side's rows that no match kept, with the other
    side's columns filled in as nulls."""
    paired = matched[tag].unique().to_list() if matched.height else []
    # `is_in([])` is not a dependable "matches nothing" in Polars -- it
    # can yield nulls, and `filter` then drops the very rows this exists
    # to keep.
    survivors = (
        preserved
        if not paired
        else preserved.filter(~pl.col(tag).is_in(paired))
    )
    missing = [column for column in columns if column not in survivors.columns]
    filled = survivors.with_columns(
        [
            pl.lit(None, dtype=matched.schema[column]).alias(column)
            for column in missing
        ]
    )
    return filled.select(columns)


def _align(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_keys: list[str],
    right_keys: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Give both sides' join keys a common dtype. Only bites when one
    side came back empty: an empty frame's columns have no values to
    infer a dtype from, and Polars refuses to join ``Null`` against
    ``Int64``."""
    for left_key, right_key in zip(left_keys, right_keys, strict=True):
        left_dtype = left.schema[left_key]
        right_dtype = right.schema[right_key]
        if left_dtype == right_dtype:
            continue
        if left_dtype == pl.Null:
            left = left.with_columns(pl.col(left_key).cast(right_dtype))
        elif right_dtype == pl.Null:
            right = right.with_columns(pl.col(right_key).cast(left_dtype))
    return left, right


# -----------------------------------------------------------------------
# GROUP BY: extract the aggregates, reduce, then project over the result
# -----------------------------------------------------------------------


def _has_aggregate(plan: _Select) -> bool:
    """Does this query reduce? A subquery's own aggregate does not count
    -- ``SELECT id, (SELECT MAX(c.id) FROM cats c) FROM items`` reduces
    nothing in the outer query, and treating it as an aggregate made the
    outer reduction look for the subquery's columns in the outer
    frame."""
    consumers = list(plan.projections)
    if plan.having is not None:
        consumers.append(plan.having)
    return any(_top_level_aggregates(node) for node in consumers)


def _match_key(node: exp.Expression) -> str:
    """A comparison key for "is this the same expression?".

    Generated SQL is not consistent about quoting -- Django writes the
    SELECT list as ``"cat" AS "cat"`` but the grouping as an ordinal,
    and hand-written MilvusQL mixes ``GROUP BY cat`` with a quoted
    projection. Matching on the rendered SQL alone made ``"cat"`` and
    ``cat`` two different expressions, so the projected column looked
    ungrouped. Quoting is stripped for the comparison only; the
    identifiers themselves stay untouched, and case still matters --
    Milvus field names are case-sensitive."""
    copied = node.copy()
    for identifier in copied.find_all(exp.Identifier):
        identifier.set("quoted", False)
    return copied.sql()


def _aggregate(frame: pl.DataFrame, ctx: _Ctx) -> pl.DataFrame:
    """Two phases, the way a SQL engine does it.

    First reduce: every distinct aggregate call in the statement becomes
    one ``agg`` output, and every ``GROUP BY`` key one group column.
    Then project: the SELECT list, ``HAVING`` and ``ORDER BY`` are
    translated again over the *reduced* frame, with grouped and
    aggregated sub-expressions substituted for the columns just
    produced. The second phase is what makes ``SUM(price) / COUNT(*)``
    and ``HAVING COUNT(*) > 2`` work -- neither is expressible inside a
    single ``agg``."""
    plan = ctx.plan
    group_names = {
        _match_key(key): f"_group{index}"
        for index, key in enumerate(plan.group)
    }
    keys = [
        _expression(key, ctx).alias(group_names[_match_key(key)])
        for key in plan.group
    ]

    aggregate_names: dict[str, str] = {}
    consumers: list[exp.Expression] = list(plan.projections)
    if plan.having is not None:
        consumers.append(plan.having)
    consumers.extend(expression for expression, _ in plan.order)
    for consumer in consumers:
        for node in _top_level_aggregates(consumer):
            aggregate_names.setdefault(
                _match_key(node), f"_agg{len(aggregate_names)}"
            )

    reductions = [
        _reduce(node, ctx).alias(name)
        for node, name in _unique_aggregates(consumers, aggregate_names)
    ]
    if keys:
        reduced = frame.group_by(keys, maintain_order=True).agg(reductions)
    else:
        # A bare aggregate with no GROUP BY is one row over everything.
        reduced = frame.select(reductions)

    def substitute(node: exp.Expression) -> pl.Expr | None:
        key = _match_key(node)
        name = group_names.get(key) or aggregate_names.get(key)
        return pl.col(name) if name is not None else None

    post = _Ctx(plan, ctx.frames, reduced).with_substitution(substitute)
    if plan.having is not None:
        reduced = reduced.filter(_expression(plan.having, post))
        post = post.against(reduced)
    if plan.order:
        reduced = _sort(reduced, post)
        post = post.against(reduced)
    return _project(reduced, post)


def _top_level_aggregates(node: exp.Expression) -> list[exp.Expression]:
    """Every aggregate call in ``node`` that is not itself nested inside
    another aggregate -- the ones that become ``agg`` outputs. Stops at a
    nested ``SELECT`` for the same reason :func:`_columns_of` does: that
    reduction belongs to the subquery, which evaluates on its own."""
    if isinstance(node, exp.AggFunc):
        return [node]
    if isinstance(node, (exp.Select, exp.Window)):
        # A windowed aggregate is not a grouped one: `SUM(x) OVER (...)`
        # keeps one row per input row, so it must not turn the query
        # into a reduction.
        return []
    found: list[exp.Expression] = []
    for child in node.args.values():
        for item in child if isinstance(child, list) else [child]:
            if isinstance(item, exp.Expression):
                found.extend(_top_level_aggregates(item))
    return found


def _unique_aggregates(
    consumers: list[exp.Expression], names: dict[str, str]
) -> list[tuple[exp.Expression, str]]:
    seen: dict[str, exp.Expression] = {}
    for consumer in consumers:
        for node in _top_level_aggregates(consumer):
            seen.setdefault(_match_key(node), node)
    return [(node, names[key]) for key, node in seen.items()]


def _reduce(node: exp.Expression, ctx: _Ctx) -> pl.Expr:
    """One aggregate call -> the Polars reduction, evaluated inside
    ``agg`` (or over the whole frame when there is no ``GROUP BY``)."""
    if isinstance(node, exp.Count):
        inner = node.this
        if isinstance(inner, exp.Distinct):
            # SQL's COUNT(DISTINCT x) ignores NULLs; `n_unique` counts
            # null as a value of its own, so drop them first.
            return (
                _expression(inner.expressions[0], ctx).drop_nulls().n_unique()
            )
        if inner is None or isinstance(inner, exp.Star):
            return pl.len()
        return _expression(inner, ctx).count()
    reduction = _AGGREGATES.get(type(node))
    if reduction is None:
        msg = f"unsupported aggregate: {node.sql()!r}"
        raise errors.NotSupportedError(msg)
    return getattr(_expression(node.this, ctx), reduction)()


# -----------------------------------------------------------------------
# Expressions
# -----------------------------------------------------------------------


def _frame_column(node: exp.Column, ctx: _Ctx) -> str:
    """A column reference -> the name it carries in the current frame.
    Qualified names are exact; a bare name matches the only source that
    has it."""
    columns = ctx.frame.columns
    if node.table:
        qualified = f"{node.table}.{node.name}"
        if qualified in columns:
            return qualified
        msg = f"unknown column {qualified!r}"
        raise errors.ProgrammingError(msg)
    if node.name in columns:
        return node.name
    matches = [
        column for column in columns if column.split(".", 1)[-1] == node.name
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        msg = f"unknown column {node.name!r}"
        raise errors.ProgrammingError(msg)
    msg = (
        f"ambiguous column {node.name!r}: it exists in "
        f"{', '.join(sorted(m.split('.', 1)[0] for m in matches))} -- "
        "qualify it with the table it comes from"
    )
    raise errors.ProgrammingError(msg)


def _expression(  # noqa: PLR0911, PLR0912 -- one return per AST node case, clearer flat than nested
    node: exp.Expression, ctx: _Ctx
) -> pl.Expr:
    """One AST node -> the Polars expression computing it against
    ``ctx.frame``."""
    if ctx.substitute is not None:
        substituted = ctx.substitute(node)
        if substituted is not None:
            return substituted
    if isinstance(node, exp.Alias):
        return _expression(node.this, ctx)
    if isinstance(node, exp.Column):
        if ctx.substitute is not None:
            # After a GROUP BY, a column that is neither a grouping key
            # nor inside an aggregate has no single value per group --
            # SQL rejects that, and so does this rather than silently
            # picking one row's value.
            msg = (
                f"column {node.sql()!r} must appear in GROUP BY or be "
                "used inside an aggregate function"
            )
            raise errors.ProgrammingError(msg)
        return pl.col(_frame_column(node, ctx))
    if (
        isinstance(node, (exp.Paren, exp.Not, exp.And, exp.Or))
        or type(node) in COMPARISON_OPS
    ):
        return _combine(node, lambda child: _expression(child, ctx))
    arithmetic = _ARITHMETIC.get(type(node))
    if arithmetic is not None:
        return getattr(_expression(node.this, ctx), arithmetic)(
            _expression(node.expression, ctx)
        )
    if isinstance(node, exp.Neg):
        return -_expression(node.this, ctx)
    if isinstance(node, (exp.Literal, exp.Boolean, exp.Null)):
        return pl.lit(resolve_value(node, {}))
    if isinstance(node, (exp.Placeholder, exp.Parameter)):
        msg = (
            f"bind parameter {node.this!r} is used in a position this "
            "engine evaluates client-side; only WHERE comparisons "
            "pushed to Milvus can bind there"
        )
        raise errors.NotSupportedError(msg)
    if isinstance(node, exp.Is):
        if isinstance(node.expression, exp.Null):
            return _expression(node.this, ctx).is_null()
        msg = f"unsupported IS comparison: {node.sql()!r}"
        raise errors.NotSupportedError(msg)
    if isinstance(node, exp.Like):
        matched = _expression(node.this, ctx).str.contains(
            _like_to_regex(str(resolve_value(node.expression, {})))
        )
        return ~matched if node.args.get("negate") else matched
    if isinstance(node, exp.Between):
        value = _expression(node.this, ctx)
        return (value >= _expression(node.args["low"], ctx)) & (
            value <= _expression(node.args["high"], ctx)
        )
    if isinstance(node, exp.In):
        return _in_expression(node, ctx)
    if isinstance(node, exp.Exists):
        return pl.lit(_subquery_frame(node.this, ctx).height > 0)
    if isinstance(node, exp.Coalesce):
        candidates = [_expression(node.this, ctx)]
        candidates.extend(
            _expression(e, ctx) for e in (node.args.get("expressions") or [])
        )
        return pl.coalesce(candidates)
    if isinstance(node, (exp.Subquery, exp.Select)):
        return pl.lit(_scalar_subquery(node, ctx))
    if isinstance(node, exp.Window):
        return _window(node, ctx)
    if isinstance(node, exp.AggFunc):
        # An aggregate reached outside a GROUP BY reduction: a HAVING or
        # ORDER BY on an ungrouped query. Reduce over the whole frame.
        return _reduce(node, ctx)
    msg = f"unsupported expression: {node.__class__.__name__} ({node.sql()!r})"
    raise errors.NotSupportedError(msg)


#: Ranking window function -> the Polars ranking method that matches
#: it. ``ROW_NUMBER`` is "ordinal" (ties broken by order of appearance),
#: ``RANK`` leaves gaps after a tie, ``DENSE_RANK`` does not.
_WINDOW_RANKS: dict[type[exp.Expression], str] = {
    exp.RowNumber: "ordinal",
    exp.Rank: "min",
    exp.DenseRank: "dense",
}


def _window(node: exp.Window, ctx: _Ctx) -> pl.Expr:
    """``<fn>() OVER (PARTITION BY ... ORDER BY ...)`` -> the Polars
    windowed expression.

    This is the shape that makes "top-k per group" expressible against a
    vector search -- ``ROW_NUMBER() OVER (PARTITION BY category ORDER BY
    distance)`` over the hits, then a filter on the rank."""
    if node.args.get("spec") is not None:
        msg = (
            "window frame clauses (ROWS/RANGE BETWEEN) are not "
            "supported: only the whole partition is available"
        )
        raise errors.NotSupportedError(msg)
    partition = [
        _expression(key, ctx) for key in node.args.get("partition_by") or []
    ]
    function = node.this
    method = _WINDOW_RANKS.get(type(function))
    if method is not None:
        windowed = _rank_expression(node, method, ctx)
    elif isinstance(function, exp.Count) or type(function) in _AGGREGATES:
        # `SUM(x) OVER (PARTITION BY g)` -- the same reduction a GROUP BY
        # would compute, broadcast back over each row of its partition.
        # Matched by type rather than by `AggFunc`, which in sqlglot also
        # covers `LAG`/`NTILE`/... -- those are window functions with no
        # reduction behind them, and belong in the error below.
        windowed = _reduce(function, ctx)
    else:
        msg = (
            f"unsupported window function: {function.sql()!r}. "
            "ROW_NUMBER, RANK, DENSE_RANK and the aggregate functions "
            "are supported"
        )
        raise errors.NotSupportedError(msg)
    # No PARTITION BY means one partition holding everything, which is
    # what a constant key gives.
    return windowed.over(partition or pl.lit(1))


def _rank_expression(node: exp.Window, method: str, ctx: _Ctx) -> pl.Expr:
    order = node.args.get("order")
    if order is None:
        msg = (
            f"{node.this.sql()!r} needs an ORDER BY inside OVER (...): "
            "a ranking has no meaning without one"
        )
        raise errors.NotSupportedError(msg)
    keys = order.expressions
    if len(keys) != 1:
        # Polars ranks one expression; ranking on several keys would
        # need a composite ordering this does not build.
        msg = (
            "a window ORDER BY is supported with one key only, got "
            f"{len(keys)}"
        )
        raise errors.NotSupportedError(msg)
    key = keys[0]
    return _expression(key.this, ctx).rank(
        method=t.cast("t.Any", method),
        descending=bool(key.args.get("desc")),
    )


def _combine(
    node: exp.Expression, translate: t.Callable[[exp.Expression], pl.Expr]
) -> pl.Expr:
    if isinstance(node, exp.Paren):
        return translate(node.this)
    if isinstance(node, exp.Not):
        return ~translate(node.this)
    if isinstance(node, exp.And):
        return translate(node.this) & translate(node.expression)
    if isinstance(node, exp.Or):
        return translate(node.this) | translate(node.expression)
    operator = {
        exp.EQ: "__eq__",
        exp.NEQ: "__ne__",
        exp.GT: "__gt__",
        exp.GTE: "__ge__",
        exp.LT: "__lt__",
        exp.LTE: "__le__",
    }[type(node)]
    return getattr(translate(node.this), operator)(translate(node.expression))


def _in_expression(node: exp.In, ctx: _Ctx) -> pl.Expr:
    query = node.args.get("query")
    if query is not None:
        # A plain Python list, not the Series itself: `is_in` treats a
        # same-dtype Series as an ambiguous element-wise comparison and
        # deprecates it (pola-rs/polars#22149).
        return _expression(node.this, ctx).is_in(
            _subquery_column(query, ctx).to_list()
        )
    if not node.expressions:
        # `col IN ()` matches nothing, same as SQL's own empty-IN-list
        # semantics and as the Milvus-side renderer's `false`.
        return pl.lit(value=False)
    return _expression(node.this, ctx).is_in(
        [resolve_value(e, {}) for e in node.expressions]
    )


def _subquery_frame(node: exp.Expression, ctx: _Ctx) -> pl.DataFrame:
    inner = node.this if isinstance(node, exp.Subquery) else node
    sub = ctx.plan.subqueries.get(id(inner))
    if sub is None:
        msg = f"unsupported subquery position: {node.sql()!r}"
        raise errors.NotSupportedError(msg)
    return _evaluate(sub, ctx.frames)


def _subquery_column(node: exp.Expression, ctx: _Ctx) -> pl.Series:
    result = _subquery_frame(node, ctx)
    if result.width != 1:
        msg = (
            "a subquery used as a value list must select exactly one "
            f"column, got {result.width}"
        )
        raise errors.ProgrammingError(msg)
    return result.to_series()


def _scalar_subquery(node: exp.Expression, ctx: _Ctx) -> t.Any:  # noqa: ANN401
    values = _subquery_column(node, ctx)
    if values.len() > 1:
        msg = "a scalar subquery returned more than one row"
        raise errors.ProgrammingError(msg)
    return values[0] if values.len() else None


def _like_to_regex(pattern: str) -> str:
    """SQL ``LIKE`` wildcards -> the regex Polars matches with. Every
    other character is escaped, so a pattern full of regex punctuation
    still means itself."""
    out = ["^"]
    for character in pattern:
        if character == "%":
            out.append(".*")
        elif character == "_":
            out.append(".")
        elif character in ".^$*+?()[]{}|\\":
            out.append(f"\\{character}")
        else:
            out.append(character)
    out.append("$")
    return "".join(out)


__all__ = [
    "build_relational_call",
    "build_set_operation_call",
    "needs_relational_engine",
]
