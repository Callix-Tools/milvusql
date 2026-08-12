"""``vector_search``/``hybrid_search`` -- integration coverage against
a real (embedded) Milvus Lite instance. Both build MilvusQL text
directly and execute it through ``connection.cursor()``, so this also
exercises ``CursorWrapper``'s translation and ``VectorField``
round-tripping live, the same non-relational bypass the module
docstring describes."""

from __future__ import annotations

import typing as t

import pytest
from milvusql_django.expressions import vector_search

from tests.integration.django.search.conftest import EMB_BOOK, Item

pytestmark = [pytest.mark.integration, pytest.mark.django]


def test_vector_search_returns_the_nearest_row_first():
    # `vector_search`'s return type is pinned to `list[Model]` in its
    # signature regardless of the `model` argument's static type --
    # cast the result, not the argument, to reach `.category`/`.rank`.
    results = t.cast(
        "list[t.Any]", vector_search(Item, "embedding", EMB_BOOK, k=5)
    )
    assert next(r.category for r in results) == "book"


def test_vector_search_applies_equality_filters_from_kwargs():
    results = t.cast(
        "list[t.Any]",
        vector_search(Item, "embedding", EMB_BOOK, k=5, category="movie"),
    )
    assert [(r.category, r.rank) for r in results] == [("movie", 5)]
