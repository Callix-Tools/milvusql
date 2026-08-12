"""``VectorField`` -- a standard Django ``Field``, not a fake-``Column``
shim (D11). It only needs to round-trip ``list[float] <-> VECTOR(n)``
DDL text; ``milvusql``'s DBAPI already carries the Python list through
as a real bind value (D1), so there is no encoding to do here either.
"""

from __future__ import annotations

import typing as t

from django.db import models

if t.TYPE_CHECKING:
    from django.db.backends.base.base import BaseDatabaseWrapper


class VectorField(models.Field):
    description = "A dense float vector (Milvus FLOAT_VECTOR)"
    empty_strings_allowed = False

    def __init__(self, dim: int | None = None, **kwargs: t.Any) -> None:
        self.dim = dim
        super().__init__(**kwargs)

    def deconstruct(
        self,
    ) -> tuple[str, str, list[t.Any], dict[str, t.Any]]:
        name, path, args, kwargs = super().deconstruct()
        if self.dim is not None:
            kwargs["dim"] = self.dim
        return name, path, args, kwargs

    def db_type(self, connection: BaseDatabaseWrapper) -> str:
        return f"VECTOR({self.dim})" if self.dim else "VECTOR"

    def get_internal_type(self) -> str:
        return "VectorField"

    def to_python(self, value: t.Any) -> list[float] | None:
        if value is None or isinstance(value, list):
            return value
        return list(value)

    def from_db_value(
        self, value: t.Any, expression: t.Any, connection: BaseDatabaseWrapper
    ) -> list[float] | None:
        return self.to_python(value)

    def get_prep_value(self, value: t.Any) -> list[float] | None:
        return self.to_python(value)


__all__ = ["VectorField"]
