"""``DatabaseOperations`` -- the handful of hooks Django's base SQL
generation needs filled in to target MilvusQL text."""

from __future__ import annotations

import decimal
import typing as t

from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import format_number


class DatabaseOperations(BaseDatabaseOperations):
    def adapt_decimalfield_value(
        self,
        value: decimal.Decimal | float | str | None,
        max_digits: int | None = None,
        decimal_places: int | None = None,
    ) -> str | None:
        """The base implementation returns ``value`` unchanged -- fine
        for a driver like psycopg2/mysqlclient that accepts a
        ``decimal.Decimal`` natively, but ``milvusql``'s DBAPI binds
        straight into a Milvus ``VARCHAR`` field (``DecimalField`` has
        no native Milvus type -- see ``data_types``), which needs the
        formatted string ``format_number`` produces, not a ``Decimal``
        object Milvus's client can't serialize."""
        if value is None:
            return None
        return format_number(value, max_digits, decimal_places)

    def quote_name(self, name: str) -> str:
        if name.startswith('"') and name.endswith('"'):
            return name
        return f'"{name}"'

    def no_limit_value(self) -> None:
        """MilvusQL's ``LIMIT`` clause can simply be omitted for "no
        limit" -- unlike sqlite (which has no such omission and uses
        ``-1``), Track A's grammar treats ``LIMIT`` as optional."""
        return

    def last_insert_id(
        self, cursor: t.Any, table_name: str, pk_name: str
    ) -> t.Any:
        """``cursor.lastrowid`` (``milvusql.dbapi``'s own PEP
        249-conventional attribute, populated from the ``ids`` Milvus's
        ``insert()`` returns for an ``auto_id`` collection -- confirmed
        directly against Milvus Lite)."""
        return cursor.lastrowid

    def sql_flush(
        self,
        style: t.Any,
        tables: list[str],
        *,
        reset_sequences: bool = False,
        allow_cascade: bool = False,
    ) -> list[str]:
        """``DELETE FROM <table>`` with no filter, for every table.
        Unverified against a real Milvus deployment (D11: this package
        is a first cut) -- flagged here rather than silently assumed
        to work, since an empty-filter ``delete()`` is a code path
        none of this project's manual verification has exercised."""
        # S608: `table` is Django's own internal table-name list
        # (model `_meta.db_table` values), not user input.
        return [
            f"DELETE FROM {self.quote_name(table)};"  # noqa: S608  # nosec B608
            for table in tables
        ]


__all__ = ["DatabaseOperations"]
