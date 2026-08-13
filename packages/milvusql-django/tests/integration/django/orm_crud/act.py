"""Integration coverage: schema-editor DDL and ORM CRUD/filtering
against a real Milvus server (via ``testcontainers``), through Django's
normal query compiler and ``CursorWrapper``'s live ``%s`` -> ``:name``
translation. ``vector_search``/``hybrid_search``'s own hand-written
MilvusQL text is covered separately, in ``tests/integration/django/
search``."""

from __future__ import annotations

import time
import typing as t

import pytest
from dj_app.models import Item as _Item
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from django.db.models import Sum

pytestmark = [pytest.mark.integration, pytest.mark.django]

#: No `django-stubs` in this workspace, so `ty` can't see the
#: metaclass-injected `objects` manager -- every call below is
#: exercised for real, just invisible statically. Same kind of
#: understood cast ``test_translate.py`` uses for ``MilvusClient``.
Item: t.Any = _Item


def _eventually(fn: t.Callable[[], t.Any], *, tries: int = 20, delay: float = 0.05) -> t.Any:
    """``TestMutation``'s own read-immediately-after-write assertions
    (not ``test_vector_field_round_trips_through_the_orm``'s -- that
    one's failure was a real, deterministic bug, `AutoField`'s 32-bit
    range silently discarding every large real ``auto_id`` pk lookup;
    see ``milvusql_django.DatabaseOperations.integer_field_ranges``'s
    docstring, fixed there). With that fixed, `self.book.save()`/
    `Item.objects.filter(...).update()` immediately followed by
    `Item.objects.get(pk=...)` in the *same* test still intermittently
    raised `Item.DoesNotExist`, even under `consistency_level=
    "Strong"` -- confirmed directly: gone on the very next retry with
    no other change, so a genuine write-visibility lag against this
    suite's own concurrently-loaded real (non-Lite) server, not a
    logic bug. `"Strong"` bounds *where* Milvus looks for the write,
    not how long the RPC that makes it visible there is still in
    flight. Never observable against Milvus Lite's single in-process
    server. A short bounded retry is the same tolerance any client of
    a real distributed system needs here."""
    for attempt in range(tries):
        try:
            return fn()
        except (AssertionError, ObjectDoesNotExist):
            if attempt == tries - 1:
                raise
            time.sleep(delay)
    msg = "unreachable"  # pragma: no cover -- the loop always returns or raises
    raise AssertionError(msg)


def test_create_model_makes_an_empty_queryable_collection(loaded_items):
    assert list(Item.objects.all()) == []


def test_vector_field_round_trips_through_the_orm(loaded_items):
    created = Item.objects.create(category="book", rank=1, embedding=[0.1] * 8)
    fetched = Item.objects.get(pk=created.pk)
    assert fetched.category == "book"
    assert fetched.rank == 1
    assert fetched.embedding == pytest.approx([0.1] * 8, abs=1e-6)


def test_add_field_succeeds_against_a_real_collection(loaded_items):
    """Against Milvus Lite, this always raised ``NotSupportedError``
    -- ``add_collection_field`` returns ``UNIMPLEMENTED`` from Lite's
    own gRPC server (a raw ``grpc.RpcError``, not even a
    ``MilvusException``), which ``milvusql``'s error translation maps
    to ``NotSupportedError``. A real Milvus server *does* implement
    ``AddCollectionField`` -- confirmed directly: the identical
    ``editor.add_field()`` call that always raised against Lite now
    succeeds outright here, and the new field is immediately visible
    through reflection."""
    tag = models.CharField(max_length=16, null=True)
    tag.set_attributes_from_name("tag")
    with loaded_items.schema_editor() as editor:
        editor.add_field(Item, tag)
    columns = {
        col.name
        for col in loaded_items.introspection.get_table_description(
            loaded_items.cursor(), Item._meta.db_table
        )
    }
    assert "tag" in columns


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

    def test_in(self):
        ranks = Item.objects.filter(rank__in=[1, 10]).values_list(
            "rank", flat=True
        )
        assert sorted(ranks) == [1, 10]

    def test_order_by(self):
        assert list(
            Item.objects.order_by("-rank").values_list("rank", flat=True)
        ) == [10, 5, 1]

    def test_count(self):
        assert Item.objects.count() == 3
        assert Item.objects.filter(category="book").count() == 1

    def test_aggregate_sum(self):
        assert Item.objects.aggregate(total=Sum("rank")) == {"total": 16}


class TestMutation:
    @pytest.fixture(autouse=True)
    def _seed(self, loaded_items):
        self.book = Item.objects.create(
            category="book", rank=1, embedding=[0.1] * 8
        )
        Item.objects.create(category="movie", rank=5, embedding=[0.9] * 8)

    def test_instance_save_updates_an_existing_row(self):
        self.book.rank = 99
        self.book.save()
        assert _eventually(lambda: Item.objects.get(pk=self.book.pk)).rank == 99
        # The row wasn't duplicated -- an upsert-by-pk, not an insert.
        assert Item.objects.count() == 2

    def test_queryset_update_changes_only_matching_rows(self):
        Item.objects.filter(category="book").update(rank=42)
        assert _eventually(lambda: Item.objects.get(pk=self.book.pk)).rank == 42
        assert Item.objects.filter(category="movie").get().rank == 5

    def test_instance_delete_removes_only_that_row(self):
        self.book.delete()
        assert list(Item.objects.values_list("category", flat=True)) == [
            "movie"
        ]
