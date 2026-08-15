"""Integration coverage for the README's full-text story through the
Django backend, against a real Milvus server:

* ``TextField`` maps to MilvusQL ``TEXT`` -- i.e. an analyzer-enabled
  field. The proof is behavioral: ``TEXT_MATCH`` (via
  ``MATCH ... AGAINST``) only works on an analyzer-enabled field, so a
  keyword filter succeeding against a schema-editor-created model
  collection is the claim verified end-to-end.
* The raw-cursor surface (Django's ``%(name)s`` placeholders through
  ``CursorWrapper``'s live translation) runs the full BM25 pipeline:
  generated ``SPARSEVEC`` DDL, BM25 index, ranked retrieval.
"""

from __future__ import annotations

import typing as t

import pytest
from django.db import models
from django.test.utils import isolate_apps
from milvusql_django.fields import VectorField
from milvusql_django.schema import create_index_and_load

pytestmark = [pytest.mark.integration, pytest.mark.django]


@pytest.fixture
def note_model(connection):
    with isolate_apps("dj_app"):

        class Note(models.Model):
            body = models.TextField()
            embedding = VectorField(dim=2)

            class Meta:
                app_label = "dj_app"

        # The cast up front: no django-stubs in this workspace, so `ty`
        # can't see the metaclass-injected `_meta`/`objects` -- same
        # understood cast the rest of the suite uses.
        note_model: t.Any = Note
        with connection.schema_editor() as editor:
            editor.create_model(note_model)
        yield note_model


def test_textfield_is_full_text_searchable(note_model, connection):
    """A ``TextField`` created through the schema editor accepts a
    ``TEXT_MATCH`` keyword filter -- which Milvus rejects outright on a
    plain (non-analyzer) VARCHAR, so this passing is the ``TextField ->
    TEXT`` mapping working end to end."""
    note_model.objects.create(
        body="milvus is a vector database", embedding=[0.1, 0.1]
    )
    note_model.objects.create(
        body="postgres is a relational database", embedding=[0.9, 0.9]
    )
    # Index + load AFTER the inserts: the keyword-match index over a
    # collection loaded *before* its rows arrive lags behind them
    # (confirmed against a real server in CI -- the same query matched
    # nothing), while loading after insert is deterministic. Same order
    # the core fulltext suite uses.
    table = note_model._meta.db_table
    create_index_and_load(
        connection,
        table,
        "embedding",
        using="FLAT",
        metric_type="L2",
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f'SELECT body FROM "{table}" WHERE MATCH(body) AGAINST (%(q)s)',
            {"q": "relational"},
        )
        rows = cursor.fetchall()
    assert rows == [("postgres is a relational database",)]


def test_bm25_pipeline_through_the_raw_cursor(connection):
    """The README's cursor example, verbatim shape: DDL with a
    BM25-generated sparse column, a BM25 index, inserts and ranked
    retrieval -- all through Django's connection and placeholder
    translation."""
    with connection.cursor() as cursor:
        cursor.execute(
            "CREATE TABLE dj_docs ("
            "id BIGINT PRIMARY KEY AUTO_INCREMENT, "
            "content TEXT, "
            "content_sparse SPARSEVEC GENERATED ALWAYS AS (BM25(content))"
            ") WITH (consistency_level='Strong')"
        )
        cursor.execute(
            "CREATE INDEX idx_dj_sparse ON dj_docs (content_sparse) "
            "USING SPARSE_INVERTED_INDEX WITH (metric_type='BM25')"
        )
        cursor.executemany(
            "INSERT INTO dj_docs (content) VALUES (%(content)s)",
            [
                {"content": "milvus is a vector database"},
                {"content": "bm25 ranks documents by keyword relevance"},
            ],
        )
        cursor.execute(
            "SELECT content FROM dj_docs "
            "ORDER BY BM25_SCORE(content_sparse, %(q)s) DESC LIMIT 1",
            {"q": "keyword relevance ranking"},
        )
        rows = cursor.fetchall()
    assert rows == [("bm25 ranks documents by keyword relevance",)]
