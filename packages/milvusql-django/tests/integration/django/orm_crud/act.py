"""Integration coverage: schema-editor DDL and ORM CRUD/filtering
against a real (embedded) Milvus Lite instance, through Django's
normal query compiler and ``CursorWrapper``'s live ``%s`` -> ``:name``
translation. ``vector_search``/``hybrid_search``'s own hand-written
MilvusQL text is covered separately, in ``tests/integration/django/
search``."""

from __future__ import annotations

import typing as t

import pytest
from dj_app.models import Item as _Item

pytestmark = [pytest.mark.integration, pytest.mark.django]

#: No `django-stubs` in this workspace, so `ty` can't see the
#: metaclass-injected `objects` manager -- every call below is
#: exercised for real, just invisible statically. Same kind of
#: understood cast ``test_translate.py`` uses for ``MilvusClient``.
Item: t.Any = _Item


def test_create_model_makes_an_empty_queryable_collection(loaded_items):
    assert list(Item.objects.all()) == []


def test_vector_field_round_trips_through_the_orm(loaded_items):
    created = Item.objects.create(category="book", rank=1, embedding=[0.1] * 8)
    fetched = Item.objects.get(pk=created.pk)
    assert fetched.category == "book"
    assert fetched.rank == 1
    assert fetched.embedding == pytest.approx([0.1] * 8, abs=1e-6)


class TestFilterLookups:
    """Only what ``DatabaseWrapper.operators`` defines: exact, gt,
    gte, lt, lte."""

    @pytest.fixture(autouse=True)
    def _seed(self, loaded_items):
        Item.objects.create(category="book", rank=1, embedding=[0.1] * 8)
        Item.objects.create(category="movie", rank=5, embedding=[0.9] * 8)
        Item.objects.create(category="game", rank=10, embedding=[0.5] * 8)

    def test_exact(self):
        assert list(
            Item.objects.filter(category="book").values_list("rank", flat=True)
        ) == [1]

    def test_gt(self):
        ranks = Item.objects.filter(rank__gt=1).values_list("rank", flat=True)
        assert sorted(ranks) == [5, 10]

    def test_gte(self):
        ranks = Item.objects.filter(rank__gte=5).values_list("rank", flat=True)
        assert sorted(ranks) == [5, 10]

    def test_lt(self):
        ranks = Item.objects.filter(rank__lt=5).values_list("rank", flat=True)
        assert sorted(ranks) == [1]

    def test_lte(self):
        ranks = Item.objects.filter(rank__lte=5).values_list("rank", flat=True)
        assert sorted(ranks) == [1, 5]
