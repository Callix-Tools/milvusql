"""Integration coverage for the ORM/schema-editor's genuinely
unsupported operations, against a real Milvus server (via
``testcontainers``).

``add_field`` used to live here too (Milvus Lite's gRPC server doesn't
implement ``AddCollectionField``, so it always raised
``NotSupportedError`` against Lite specifically) -- a real server
*does* implement it, so that's now a happy-path test in ``act.py``
instead (``test_add_field_succeeds_against_a_real_collection``).
``remove_field``/``alter_field`` are different in kind: Milvus itself
has no DROP FIELD or ALTER COLUMN capability at all (same restriction
``sqlglot-milvus`` enforces at the SQL grammar level), so
``DatabaseSchemaEditor`` raises ``NotImplementedError`` directly in
Python, before any RPC -- true regardless of Lite vs. a real server,
and exercised here without needing ``loaded_items`` at all."""

from __future__ import annotations

import typing as t

import pytest
from dj_app.models import Item as _Item
from django.db import models

pytestmark = [pytest.mark.integration, pytest.mark.django]

#: No `django-stubs` in this workspace, so `ty` can't see the
#: metaclass-injected `objects` manager -- every call below is
#: exercised for real, just invisible statically. Same kind of
#: understood cast ``test_translate.py`` uses for ``MilvusClient``.
Item: t.Any = _Item


def test_remove_field_raises_not_implemented(connection):
    tag = models.CharField(max_length=16, null=True)
    tag.set_attributes_from_name("tag")
    with (
        connection.schema_editor() as editor,
        pytest.raises(NotImplementedError, match="cannot drop a field"),
    ):
        editor.remove_field(Item, tag)


def test_alter_field_raises_not_implemented(connection):
    old = models.CharField(max_length=64)
    old.set_attributes_from_name("category")
    new = models.CharField(max_length=128)
    new.set_attributes_from_name("category")
    with (
        connection.schema_editor() as editor,
        pytest.raises(NotImplementedError, match="cannot change a field"),
    ):
        editor.alter_field(Item, old, new)
