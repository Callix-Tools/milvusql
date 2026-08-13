"""Shared seed data for the ``search`` action's ``act.py``/
``error.py``: both need the same two-row ``Item`` collection, seeded
against a real Milvus server (via ``testcontainers``) via
``loaded_items``."""

from __future__ import annotations

import typing as t

import pytest
from dj_app.models import Item as _Item

#: No `django-stubs` in this workspace, so `ty` can't see the
#: metaclass-injected `objects` manager -- every call below is
#: exercised for real, just invisible statically. Same kind of
#: understood cast ``test_translate.py`` uses for ``MilvusClient``.
Item: t.Any = _Item

EMB_BOOK = [0.1] * 8
EMB_MOVIE = [0.9] * 8


@pytest.fixture(autouse=True)
def _seed(loaded_items):
    Item.objects.create(category="book", rank=1, embedding=EMB_BOOK)
    Item.objects.create(category="movie", rank=5, embedding=EMB_MOVIE)
