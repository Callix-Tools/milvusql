"""Integration coverage for the ORM/schema-editor's one documented
unsupported operation, against a real (embedded) Milvus Lite instance."""

from __future__ import annotations

import typing as t

import pytest
from dj_app.models import Item as _Item
from django.db import NotSupportedError, models

pytestmark = [pytest.mark.integration, pytest.mark.django]

#: No `django-stubs` in this workspace, so `ty` can't see the
#: metaclass-injected `objects` manager -- every call below is
#: exercised for real, just invisible statically. Same kind of
#: understood cast ``test_translate.py`` uses for ``MilvusClient``.
Item: t.Any = _Item


def test_add_field_raises_not_supported_against_a_real_collection(
    loaded_items,
):
    """Documents current, real behavior rather than forcing a happy
    path: ``milvusql`` core's ``ALTER TABLE`` dispatch
    (``ast_to_pymilvus.build_call``) is unconditionally unimplemented,
    so ``add_field`` always raises here -- despite ``schema.py``'s
    own module docstring listing ``AddField`` under "What works"."""
    tag = models.CharField(max_length=16, null=True)
    tag.set_attributes_from_name("tag")
    with (
        loaded_items.schema_editor() as editor,
        pytest.raises(
            NotSupportedError, match="ALTER TABLE is not implemented"
        ),
    ):
        editor.add_field(Item, tag)
