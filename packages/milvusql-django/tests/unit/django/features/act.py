"""``DatabaseFeatures`` -- unit coverage for the "honest empty/False"
flags (D11) other Django internals read directly, e.g. the schema
editor's ``atomic_migration = connection.features.can_rollback_ddl
and atomic`` check (``BaseDatabaseSchemaEditor.__init__``)."""

from __future__ import annotations

import pytest
from milvusql_django.features import DatabaseFeatures

pytestmark = [pytest.mark.unit, pytest.mark.django]


def test_transactional_and_constraint_features_are_all_switched_off():
    features = DatabaseFeatures(connection=None)
    assert (
        features.supports_transactions,
        features.uses_savepoints,
        features.supports_foreign_keys,
        features.can_rollback_ddl,
        features.has_select_for_update,
        features.supports_ignore_conflicts,
    ) == (False, False, False, False, False, False)
