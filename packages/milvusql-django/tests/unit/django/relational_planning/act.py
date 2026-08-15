"""The SQL Django's compiler emits for joins, aggregations and
subqueries has to be SQL the DBAPI's relational planner accepts.

None of this is backend code -- the planner lives in `milvusql` and this
backend inherits the capability. What these pin down is Django's own
spelling, which is not the one a person writes: every column is quoted
and table-qualified, every ``.values(...).annotate(...)`` groups by
*ordinal position* rather than by name, and ``ORDER BY`` does too.
Offline: compiling a queryset and planning a ``Call`` both do zero
network I/O."""

from __future__ import annotations

import typing as t
from types import SimpleNamespace

import pytest
import sqlglot
from django.db import models
from django.db.models import Avg, Count, Exists, OuterRef, Subquery
from django.test.utils import isolate_apps
from pymilvus import MilvusClient

import milvusql
from milvusql.translate.ast_to_pymilvus import build_call

pytestmark = [pytest.mark.unit, pytest.mark.django]

_client = t.cast("MilvusClient", MilvusClient)


@pytest.fixture
def related():
    """A model pair with a real relation, registered only for the test.

    ``dj_app.models`` deliberately has no ``ForeignKey`` -- adding one
    would change the collection every other test in this package
    creates from that app. ``isolate_apps`` keeps these two out of the
    permanent registry, so the migration tests still see exactly the
    app they expect."""
    with isolate_apps("dj_app"):

        class Category(models.Model):
            title = models.CharField(max_length=64)

            class Meta:
                app_label = "dj_app"

        class Item(models.Model):
            category = models.ForeignKey(
                Category, on_delete=models.CASCADE, db_constraint=False
            )
            rank = models.IntegerField()

            class Meta:
                app_label = "dj_app"

        yield SimpleNamespace(category=Category, item=Item)


def _plan(queryset):
    sql, params = queryset.query.sql_with_params()
    sql = sql % tuple(repr(p) for p in params)
    return sql, build_call(_client, sqlglot.parse_one(sql, read="milvus"), {})


def _chain(call, raws):
    calls = [call]
    for index in range(1, len(raws)):
        call = call.then(raws[index - 1])
        calls.append(call)
    return calls, call.postprocess(raws[-1])


class TestAggregation:
    def test_values_annotate_groups_by_ordinal(self, related):
        """Django never names the grouping column -- ``GROUP BY 1``.
        Left unresolved that groups by the constant 1, i.e. one group
        for the whole collection."""
        sql, call = _plan(
            related.item.objects.values("category_id").annotate(n=Count("id"))
        )
        assert "GROUP BY 1" in sql
        rows, desc, _rc, _l = call.postprocess(
            [
                {"category_id": 1, "id": 10},
                {"category_id": 1, "id": 11},
                {"category_id": 2, "id": 12},
            ]
        )
        assert rows == [(1, 2), (2, 1)]
        assert [column[0] for column in desc] == ["category_id", "n"]

    def test_annotate_with_an_ordered_average(self, related):
        sql, call = _plan(
            related.item.objects.values("category_id")
            .annotate(a=Avg("rank"))
            .order_by("-a")
        )
        assert "ORDER BY 2 DESC" in sql
        rows, _desc, _rc, _l = call.postprocess(
            [
                {"category_id": 1, "rank": 1},
                {"category_id": 1, "rank": 3},
                {"category_id": 2, "rank": 10},
            ]
        )
        assert rows == [(2, 10.0), (1, 2.0)]

    def test_annotate_then_filter_becomes_having(self, related):
        sql, call = _plan(
            related.item.objects.values("category_id")
            .annotate(n=Count("id"))
            .filter(n__gt=1)
        )
        assert "HAVING" in sql
        rows, _desc, _rc, _l = call.postprocess(
            [
                {"category_id": 1, "id": 10},
                {"category_id": 1, "id": 11},
                {"category_id": 2, "id": 12},
            ]
        )
        assert rows == [(1, 2)]


class TestRelations:
    def test_filtering_across_a_relation_plans_both_reads(self, related):
        sql, call = _plan(
            related.item.objects.filter(category__title="book").values("id")
        )
        assert "INNER JOIN" in sql
        calls, (rows, _d, _rc, _l) = _chain(
            call,
            [
                [{"id": 5}],
                [{"id": 1, "category_id": 5}, {"id": 2, "category_id": 9}],
            ],
        )
        collections = [c.kwargs["collection_name"] for c in calls]
        assert collections == ["dj_app_category", "dj_app_item"]
        assert rows == [(1,)]

    def test_the_filter_on_the_related_table_reaches_milvus(self, related):
        _sql, call = _plan(
            related.item.objects.filter(category__title="book").values("id")
        )
        assert call.kwargs["filter"] == 'title == "book"'

    def test_an_in_subquery_plans_both_reads(self, related):
        sql, call = _plan(
            related.item.objects.filter(
                category_id__in=related.category.objects.filter(
                    title="book"
                ).values("id")
            ).values("id")
        )
        assert "IN (SELECT" in sql
        calls, (rows, _d, _rc, _l) = _chain(
            call,
            [
                [{"id": 5}],
                [{"id": 1, "category_id": 5}, {"id": 2, "category_id": 9}],
            ],
        )
        assert len(calls) == 2
        assert rows == [(1,)]


class TestExists:
    def test_exists_with_outerref_plans_a_semi_join(self, related):
        """``Exists(... OuterRef(...))`` is the correlated-``EXISTS``
        shape the planner decorrelates into a semi join -- Django's own
        spelling of it (aliased subquery, ``LIMIT 1``, everything
        quoted) plans into one read per table and keeps exactly the
        outer rows with a match."""
        sql, call = _plan(
            related.item.objects.filter(
                Exists(
                    related.category.objects.filter(pk=OuterRef("category_id"))
                )
            ).values("id")
        )
        assert "EXISTS" in sql
        calls, (rows, _d, _rc, _l) = _chain(
            call,
            [
                [{"id": 5}],
                [{"id": 1, "category_id": 5}, {"id": 2, "category_id": 9}],
            ],
        )
        collections = [c.kwargs["collection_name"] for c in calls]
        assert collections == ["dj_app_category", "dj_app_item"]
        assert rows == [(1,)]

    def test_negated_exists_is_the_anti_join(self, related):
        _sql, call = _plan(
            related.item.objects.filter(
                ~Exists(
                    related.category.objects.filter(pk=OuterRef("category_id"))
                )
            ).values("id")
        )
        _calls, (rows, _d, _rc, _l) = _chain(
            call,
            [
                [{"id": 5}],
                [{"id": 1, "category_id": 5}, {"id": 2, "category_id": 9}],
            ],
        )
        assert rows == [(2,)]


class TestUnsupported:
    def test_a_correlated_scalar_annotation_is_rejected_by_name(self, related):
        """``annotate(Subquery(... OuterRef(...)))`` puts a correlated
        subquery in the SELECT list -- a value per outer row, which the
        semi-join rewrite cannot express. The error still names the
        construct rather than complaining about an unknown table."""
        with pytest.raises(milvusql.NotSupportedError, match="correlated"):
            _plan(
                related.item.objects.annotate(
                    title=Subquery(
                        related.category.objects.filter(
                            pk=OuterRef("category_id")
                        ).values("title")[:1]
                    )
                ).values("id", "title")
            )
