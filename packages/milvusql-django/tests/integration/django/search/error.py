"""``vector_search``/``hybrid_search`` -- integration coverage for
their error paths against a real (embedded) Milvus Lite instance."""

from __future__ import annotations

import pytest
from django.db import NotSupportedError
from milvusql_django.expressions import hybrid_search, vector_search

from tests.integration.django.search.conftest import EMB_BOOK, Item

pytestmark = [pytest.mark.integration, pytest.mark.django]


def test_vector_search_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        vector_search(Item, "embedding", EMB_BOOK, metric="euclidean")


def test_hybrid_search_raises_not_supported_against_a_real_collection():
    """Documents current, real behavior rather than forcing a happy
    path: unlike ``vector_search``, ``HYBRID SEARCH`` dispatch is
    unconditionally unimplemented in ``ast_to_pymilvus.build_call``."""
    with pytest.raises(
        NotSupportedError, match="HYBRID SEARCH is not implemented"
    ):
        hybrid_search(Item, [("embedding", "cosine", EMB_BOOK, 1.0)], k=5)
