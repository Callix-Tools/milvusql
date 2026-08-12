"""``VECTOR``/``SPARSEVEC`` column types and their ``Comparator``.

Deliberately parallels ``pgvector.sqlalchemy`` almost line for line
(the design plan's own D3-adjacent note): the only real difference is
that ``self.op("<->", ...)`` here generates text a *custom* compiler
(``milvusql_sqlalchemy``'s, backed by ``sqlglot-milvus``) understands,
not a native PostgreSQL operator. ``ColumnOperators.op()`` builds a
``custom_op`` SQLAlchemy's base ``SQLCompiler`` already knows how to
render as plain infix text (``left <-> right``) -- no compiler-side
code needed for the operators themselves, only for the ``VECTOR(n)``
column type in DDL (``compiler.py``).

Bind values are never stringified here: ``bind_processor`` is the
identity function, because ``milvusql``'s DBAPI (D1) takes real Python
values in its ``parameters`` dict and never serializes a vector into
SQL text -- there is nothing for this type to encode.
"""

from __future__ import annotations

import typing as t

from sqlalchemy.types import Float, UserDefinedType

if t.TYPE_CHECKING:
    from sqlalchemy.engine import Dialect
    from sqlalchemy.sql.type_api import (
        _BindProcessorType,
        _ResultProcessorType,
    )


class VECTOR(UserDefinedType):
    """``VECTOR(n)`` -- a dense float vector column, MilvusQL's
    ``FLOAT_VECTOR`` field type."""

    cache_ok = True

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def get_col_spec(self, **kw: object) -> str:
        return f"VECTOR({self.dim})" if self.dim else "VECTOR"

    def bind_processor(
        self, dialect: Dialect
    ) -> _BindProcessorType[list[float] | None]:
        def process(value: list[float] | None) -> list[float] | None:
            return list(value) if value is not None else None

        return process

    def result_processor(
        self, dialect: Dialect, coltype: object
    ) -> _ResultProcessorType[list[float] | None]:
        def process(value: list[float] | None) -> list[float] | None:
            return list(value) if value is not None else None

        return process

    class comparator_factory(UserDefinedType.Comparator):  # noqa: N801 -- exact name SQLAlchemy requires
        """Spelled exactly as Track A's distance operators (its own
        README: "spelled exactly as pgvector spells them"), so this
        mirrors pgvector's comparator one for one."""

        def l2_distance(self, other: object) -> t.Any:
            return self.op("<->", return_type=Float)(other)

        def max_inner_product(self, other: object) -> t.Any:
            return self.op("<#>", return_type=Float)(other)

        def cosine_distance(self, other: object) -> t.Any:
            return self.op("<=>", return_type=Float)(other)

        def l1_distance(self, other: object) -> t.Any:
            return self.op("<+>", return_type=Float)(other)


class SPARSEVEC(UserDefinedType):
    """``SPARSEVEC`` -- a sparse float vector column, MilvusQL's
    ``SPARSE_FLOAT_VECTOR`` field type. Only ``max_inner_product`` is
    exposed: Milvus's sparse index only supports the ``IP`` metric."""

    cache_ok = True

    def get_col_spec(self, **kw: object) -> str:
        return "SPARSEVEC"

    class comparator_factory(UserDefinedType.Comparator):  # noqa: N801 -- exact name SQLAlchemy requires
        def max_inner_product(self, other: object) -> t.Any:
            return self.op("<#>", return_type=Float)(other)


__all__ = ["SPARSEVEC", "VECTOR"]
