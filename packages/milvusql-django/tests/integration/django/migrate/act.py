"""Integration coverage for Django's own migration bookkeeping
(``django.db.migrations.recorder.MigrationRecorder``) against a real
(embedded) Milvus Lite instance -- regression coverage for the bug its
``django_migrations`` table used to crash on.

Not exercised through ``call_command("migrate")`` directly: Django
itself skips ``ensure_schema()`` whenever the computed migration plan
is empty (confirmed directly, ``MigrationExecutor.migrate``: "don't
create [the table] if there are no migrations to apply"), and
``dj_app`` (this suite's only installed app) carries no migrations
directory -- its plan is always empty, so a bare ``call_command
("migrate")`` here would exercise Django's own early-return, not this
backend's schema creation at all. Calling ``ensure_schema()`` directly
is what a project *with* real migrations actually triggers on its
first ``migrate``.

``MigrationRecorder.Migration`` has an ``applied DateTimeField`` --
with no ``DateTimeField`` entry in ``data_types`` at all,
``Field.db_type()`` returned ``None`` and
``DatabaseSchemaEditor._column_definition`` blew up with a bare
``TypeError`` before a single user migration could ever run. Fixing
that surfaced a second, deeper one: Milvus refuses ``CREATE TABLE``
outright for a schema with zero vector fields, which
``django_migrations`` -- app/name/id/applied, no vector column among
them -- is a textbook example of. Both are covered here."""

from __future__ import annotations

import pytest
from django.db.migrations.recorder import MigrationRecorder

pytestmark = [pytest.mark.integration, pytest.mark.django]


def test_ensure_schema_creates_the_recorder_table(connection):
    recorder = MigrationRecorder(connection)
    recorder.ensure_schema()
    assert recorder.has_table()


def test_ensure_schema_is_idempotent(connection):
    recorder = MigrationRecorder(connection)
    recorder.ensure_schema()
    recorder.ensure_schema()  # must not raise a second time
    assert recorder.has_table()


def test_recording_an_applied_migration_round_trips(connection):
    """Not just DDL -- the recorder's own ``INSERT`` (``applied``'s
    ``DateTimeField`` value included) has to round-trip too."""
    recorder = MigrationRecorder(connection)
    recorder.ensure_schema()
    recorder.record_applied("dj_app", "0001_initial")
    assert ("dj_app", "0001_initial") in recorder.applied_migrations()
