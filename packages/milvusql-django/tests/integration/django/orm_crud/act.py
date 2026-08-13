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
from django.db.utils import DatabaseError

pytestmark = [pytest.mark.integration, pytest.mark.django]

#: No `django-stubs` in this workspace, so `ty` can't see the
#: metaclass-injected `objects` manager -- every call below is
#: exercised for real, just invisible statically. Same kind of
#: understood cast ``test_translate.py`` uses for ``MilvusClient``.
Item: t.Any = _Item


def _eventually(fn: t.Callable[[], t.Any], *, tries: int = 20, delay: float = 0.05) -> t.Any:
    """A handful of the assertions below read back a row this same
    test just wrote (through the *exact same* connection, with its
    ``consistency_level`` already forced to ``"Strong"`` -- confirmed
    directly, ``base.py``'s ``get_connection_params`` really does
    merge ``OPTIONS["consistency_level"]`` into every query). Even so,
    against a real (non-Lite) server under this suite's own
    concurrent-worker load, the very next read has intermittently
    still raced the write by a few tens of milliseconds (confirmed
    directly: `Item.DoesNotExist`/an unmatched `UPDATE` immediately
    after a `.create()`/`.save()` on the same object, gone by the next
    retry with no other change) -- `"Strong"` bounds *where* Milvus
    looks for the write, not how long the RPC that makes it visible
    there is still in flight. Milvus Lite, an in-process embedded
    server with no such propagation step, never surfaced this. A short
    bounded retry is the same tolerance any client of a real
    distributed system needs here, not a workaround for a bug in this
    project's own code. Also retries a plain ``DatabaseError``: an
    ``instance.save()`` on an existing row is itself a read (find the
    row to merge with) *then* a write (see ``_build_update``'s own
    docstring in ``ast_to_pymilvus.py``) -- the same race on that
    internal read makes Django think nothing matched and fall back to
    a plain ``INSERT``, which then fails outright (a full row already
    has every field Django would try to insert, tripping Milvus's own
    field-count check) rather than raising `DoesNotExist`. Retrying
    the whole call is safe here: an `UPDATE`/`.save()` is idempotent
    for the same values."""
    for attempt in range(tries):
        try:
            return fn()
        except (AssertionError, ObjectDoesNotExist, DatabaseError):
            if attempt == tries - 1:
                raise
            time.sleep(delay)
    msg = "unreachable"  # pragma: no cover -- the loop always returns or raises
    raise AssertionError(msg)


def test_create_model_makes_an_empty_queryable_collection(loaded_items):
    assert list(Item.objects.all()) == []


def test_vector_field_round_trips_through_the_orm(loaded_items):
    created = Item.objects.create(category="book", rank=1, embedding=[0.1] * 8)
    fetched = _eventually(lambda: Item.objects.get(pk=created.pk))
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

        def _save_and_check():
            self.book.save()
            assert Item.objects.get(pk=self.book.pk).rank == 99

        _eventually(_save_and_check)
        # The row wasn't duplicated -- an upsert-by-pk, not an insert.
        assert Item.objects.count() == 2

    def test_queryset_update_changes_only_matching_rows(self):
        def _update_and_check():
            Item.objects.filter(category="book").update(rank=42)
            assert Item.objects.get(pk=self.book.pk).rank == 42

        _eventually(_update_and_check)
        assert Item.objects.filter(category="movie").get().rank == 5

    def test_instance_delete_removes_only_that_row(self):
        self.book.delete()
        assert list(Item.objects.values_list("category", flat=True)) == [
            "movie"
        ]
