"""Integration coverage for ``Exists(... OuterRef(...))`` -- Django's
correlated-``EXISTS`` construct -- through the ORM against a real
Milvus server, decorrelated into a semi/anti join by the DBAPI's
planner."""

from __future__ import annotations

import typing as t
from types import SimpleNamespace

import pytest
from django.db import models
from django.db.models import Exists, OuterRef
from django.test.utils import isolate_apps
from milvusql_django.fields import VectorField
from milvusql_django.schema import create_index_and_load

pytestmark = [pytest.mark.integration, pytest.mark.django]


@pytest.fixture
def family(connection):
    with isolate_apps("dj_app"):

        class Parent(models.Model):
            name = models.CharField(max_length=32)
            embedding = VectorField(dim=2)

            class Meta:
                app_label = "dj_app"

        class Child(models.Model):
            parent = models.ForeignKey(
                Parent, on_delete=models.CASCADE, db_constraint=False
            )
            active = models.BooleanField()
            embedding = VectorField(dim=2)

            class Meta:
                app_label = "dj_app"

        # The casts up front: no django-stubs in this workspace, so
        # `ty` can't see the metaclass-injected `_meta`/`objects` --
        # same understood cast the rest of the suite uses.
        parent_model: t.Any = Parent
        child_model: t.Any = Child
        with connection.schema_editor() as editor:
            editor.create_model(parent_model)
            editor.create_model(child_model)
        for model in (parent_model, child_model):
            create_index_and_load(
                connection,
                model._meta.db_table,
                "embedding",
                using="FLAT",
                metric_type="L2",
            )

        with_active = parent_model.objects.create(
            name="with-active", embedding=[0.1, 0.1]
        )
        with_inactive = parent_model.objects.create(
            name="with-inactive", embedding=[0.2, 0.2]
        )
        parent_model.objects.create(name="childless", embedding=[0.3, 0.3])
        child_model.objects.create(
            parent=with_active, active=True, embedding=[0.0, 0.0]
        )
        child_model.objects.create(
            parent=with_inactive, active=False, embedding=[0.0, 0.0]
        )
        yield SimpleNamespace(parent=parent_model, child=child_model)


def test_exists_with_outerref(family):
    names = set(
        family.parent.objects.filter(
            Exists(family.child.objects.filter(parent_id=OuterRef("pk")))
        ).values_list("name", flat=True)
    )
    assert names == {"with-active", "with-inactive"}


def test_exists_with_an_inner_filter(family):
    names = set(
        family.parent.objects.filter(
            Exists(
                family.child.objects.filter(
                    parent_id=OuterRef("pk"), active=True
                )
            )
        ).values_list("name", flat=True)
    )
    assert names == {"with-active"}


def test_negated_exists_is_the_anti_join(family):
    names = set(
        family.parent.objects.filter(
            ~Exists(family.child.objects.filter(parent_id=OuterRef("pk")))
        ).values_list("name", flat=True)
    )
    assert names == {"childless"}
