"""``DatabaseOperations`` -- the handful of hooks Django's base SQL
generation needs filled in to target MilvusQL text."""

from __future__ import annotations

import typing as t

from django.db.backends.base.operations import BaseDatabaseOperations


class DatabaseOperations(BaseDatabaseOperations):
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
