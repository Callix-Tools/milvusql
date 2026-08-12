"""``DatabaseFeatures`` -- turns off what Milvus genuinely doesn't
have (D11: honest empty/False, not a silent no-op pretending to work).
"""

from __future__ import annotations

from django.db.backends.base.features import BaseDatabaseFeatures


class DatabaseFeatures(BaseDatabaseFeatures):
    supports_transactions = False
    uses_savepoints = False
    supports_foreign_keys = False
    supports_column_check_constraints = False
    supports_table_check_constraints = False
    can_rollback_ddl = False
    supports_default_values = False
    supports_empty_insert = False
    supports_select_for_update = False
    supports_select_for_update_with_limit = False
    supports_ignore_conflicts = False
    supports_paramstyle_pyformat = False
    test_db_allows_multiple_connections = False
    has_select_for_update = False


__all__ = ["DatabaseFeatures"]
