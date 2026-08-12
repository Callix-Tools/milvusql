"""``DatabaseSchemaEditor`` -- unit coverage for the alterations Milvus
can do neither of: the collection needs recreating, same restriction
``sqlglot-milvus`` enforces at the SQL level. Uses the same
``connection``/``item_model`` fixtures as this action's ``act.py``,
shared via this directory's ``conftest.py``."""

from __future__ import annotations

import pytest
from milvusql_django.schema import DatabaseSchemaEditor

pytestmark = [pytest.mark.unit, pytest.mark.django]


class TestUnsupportedAlterations:
    def test_remove_field_raises_not_implemented(self, connection, item_model):
        editor = DatabaseSchemaEditor(connection, collect_sql=True)
        category = item_model._meta.local_fields[1]
        with pytest.raises(NotImplementedError, match="cannot drop a field"):
            editor.remove_field(item_model, category)

    def test_alter_field_raises_not_implemented(self, connection, item_model):
        editor = DatabaseSchemaEditor(connection, collect_sql=True)
        category = item_model._meta.local_fields[1]
        with pytest.raises(NotImplementedError, match="cannot change a field"):
            editor.alter_field(item_model, category, category)
