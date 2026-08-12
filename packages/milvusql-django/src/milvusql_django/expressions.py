"""Vector and hybrid search -- explicit helpers, not queryset methods
(D11). Django's ``SQLCompiler`` has no concept of an ANN ``ORDER BY``
or a ``HYBRID SEARCH`` clause, and teaching it one is exactly the
Model-metaclass-forking path ``django-cassandra-engine`` took and this
package deliberately didn't (see the design plan's D11). These build
MilvusQL text directly and execute it through ``connection.cursor()``,
the same non-negotiable bypass -- scoped to only the parts of MilvusQL
that are genuinely not relational, not the whole query surface (plain
``Model.objects.filter(...)`` still goes through Django's normal
compiler, untouched).
"""

from __future__ import annotations

import typing as t

from django.db import connections

if t.TYPE_CHECKING:
    from django.db.models import Model

#: Track A's distance operators (its own README: "spelled exactly as
#: pgvector spells them").
_METRIC_OPS = {
    "l2": "<->",
    "cosine": "<=>",
    "inner_product": "<#>",
    "l1": "<+>",
}


def _quote(name: str) -> str:
    return f'"{name}"'


def _build_filter(
    filters: dict[str, t.Any],
) -> tuple[str, dict[str, t.Any]]:
    if not filters:
        return "", {}
    clauses = []
    params = {}
    for i, (column, value) in enumerate(filters.items()):
        name = f"filter_{i}"
        clauses.append(f"{_quote(column)} = :{name}")
        params[name] = value
    return " WHERE " + " AND ".join(clauses), params


def _rows_to_instances(
    model: type[t.Any], cursor: t.Any, using: str
) -> list[Model]:
    columns = [col[0] for col in cursor.description]
    return [model.from_db(using, columns, row) for row in cursor.fetchall()]


def vector_search(  # noqa: PLR0913 -- each param is a meaningful, named search option
    model: type[t.Any],
    field_name: str,
    query_vector: list[float],
    *,
    k: int = 10,
    metric: str = "cosine",
    using: str = "default",
    **filters: t.Any,
) -> list[Model]:
    """
    ``vector_search(Item, "embedding", query_vec, k=5, category="book")``
    """
    op = _METRIC_OPS.get(metric)
    if op is None:
        msg = (
            f"unknown metric {metric!r}, expected one of {sorted(_METRIC_OPS)}"
        )
        raise ValueError(msg)

    table = model._meta.db_table
    columns = [f.column for f in model._meta.local_fields]
    select_list = ", ".join(_quote(c) for c in columns)
    where_sql, where_params = _build_filter(filters)

    # S608: every interpolated piece is a quoted identifier from
    # `model._meta` (developer-defined field/table names, not runtime
    # user input) or a fixed operator from `_METRIC_OPS`; every actual
    # value (query_vector, filter values, k) is a `:name` bind param,
    # never inlined -- same invariant `milvusql` core's own filter
    # renderer documents (D1).
    sql = (
        f"SELECT {select_list} FROM {_quote(table)}{where_sql} "  # noqa: S608  # nosec B608
        f"ORDER BY {_quote(field_name)} {op} :query_vector LIMIT :limit"
    )
    params = {**where_params, "query_vector": query_vector, "limit": k}

    connection = connections[using]
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return _rows_to_instances(model, cursor, using)


def hybrid_search(
    model: type[t.Any],
    arms: list[tuple[str, str, list[float], float]],
    *,
    k: int = 10,
    rerank: str = "RRF",
    using: str = "default",
    **filters: t.Any,
) -> list[Model]:
    """
    ``hybrid_search(Item, [
        ("embedding", "cosine", dense_q, 0.7),
        ("sparse", "inner_product", sparse_q, 0.3),
    ], k=10)``
    """
    table = model._meta.db_table
    columns = [f.column for f in model._meta.local_fields]
    select_list = ", ".join(_quote(c) for c in columns)
    where_sql, where_params = _build_filter(filters)

    params = dict(where_params)
    arm_sql_parts = []
    for i, (field_name, metric, vector, weight) in enumerate(arms):
        op = _METRIC_OPS.get(metric)
        if op is None:
            msg = (
                f"unknown metric {metric!r}, "
                f"expected one of {sorted(_METRIC_OPS)}"
            )
            raise ValueError(msg)
        name = f"arm_{i}"
        params[name] = vector
        arm_sql_parts.append(
            f"{_quote(field_name)} {op} :{name} WEIGHT {weight}"
        )
    params["limit"] = k

    # S608: same reasoning as vector_search() above -- identifiers and
    # a fixed operator set, every value a bind param.
    sql = (
        f"SELECT {select_list} FROM {_quote(table)}{where_sql} "  # noqa: S608  # nosec B608
        f"HYBRID SEARCH ({', '.join(arm_sql_parts)}) RERANK {rerank} "
        f"LIMIT :limit"
    )

    connection = connections[using]
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return _rows_to_instances(model, cursor, using)


__all__ = ["hybrid_search", "vector_search"]
