"""``DatabaseSchemaEditor`` -- unit coverage using Django's built-in
``collect_sql=True`` mode: the SQL that would run is appended to
``editor.collected_sql`` instead of executed, so ``create_model``/
``add_field``'s generated text is verifiable with zero I/O. Fields
are real ``django.db.models.Field`` instances given attributes via
``set_attributes_from_name`` (done by the ``item_model`` fixture in
this action's ``conftest.py``) -- the one piece a bare ``Field()``
doesn't get until a model's metaclass attaches it (which needs
``django.setup()``), and the only piece ``_column_definition``
actually reads; the model itself is a minimal duck-typed stand-in
exposing just ``_meta.db_table``/``local_fields``/``auto_field``."""

from __future__ import annotations

import pytest
from django.db import models
from milvusql_django.schema import DatabaseSchemaEditor

from tests.unit.django.schema_editor.conftest import _named

pytestmark = [pytest.mark.unit, pytest.mark.django]


class TestCreateModel:
    def test_generates_one_create_table_statement_with_every_column(
        self, connection, item_model
    ):
        editor = DatabaseSchemaEditor(connection, collect_sql=True)
        editor.create_model(item_model)
        assert editor.collected_sql == [
            'CREATE TABLE "items" ("id" BIGINT PRIMARY KEY AUTO_INCREMENT, '
            '"category" VARCHAR(64), "embedding" VECTOR(8));'
        ]


class TestAddField:
    def test_generates_one_alter_table_add_field_statement(
        self, connection, item_model
    ):
        editor = DatabaseSchemaEditor(connection, collect_sql=True)
        tag = _named(models.CharField(max_length=32, null=True), "tag")
        editor.add_field(item_model, tag)
        assert editor.collected_sql == [
            'ALTER TABLE "items" ADD FIELD "tag" VARCHAR(32);'
        ]
