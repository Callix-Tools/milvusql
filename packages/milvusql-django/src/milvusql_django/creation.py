"""``DatabaseCreation`` -- deliberately thin (D11 spike scope).

Django's default ``_create_test_db``/``_destroy_test_db`` assume a SQL
``CREATE DATABASE``/``DROP DATABASE`` a nodb connection can issue.
Milvus's closest equivalent is ``MilvusClient.create_database()``, a
client call, not SQL text -- out of scope for this first cut (see the
design plan's D11: the schema/migration layer is a spike, not a
finished implementation). Overridden here as explicit no-ops so
running the test runner doesn't crash with a confusing SQL error; a
real implementation is future work, not a silent gap someone would hit
without a clear signal first.
"""

from __future__ import annotations

from django.db.backends.base.creation import BaseDatabaseCreation


class DatabaseCreation(BaseDatabaseCreation):
    def _create_test_db(
        self, verbosity: int, autoclobber: bool, keepdb: bool = False
    ) -> str:
        return self._get_test_db_name()

    def _destroy_test_db(
        self, test_database_name: str, verbosity: int
    ) -> None:
        pass

    def sql_table_creation_suffix(self) -> str:
        return ""


__all__ = ["DatabaseCreation"]
