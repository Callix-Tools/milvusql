"""``vector_search``/``hybrid_search`` -- integration coverage for
their error paths against a real Milvus server (via ``testcontainers``)."""

from __future__ import annotations

import pytest
from milvusql_django.expressions import vector_search

from tests.integration.django.search.conftest import EMB_BOOK, Item

pytestmark = [pytest.mark.integration, pytest.mark.django]


def test_vector_search_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        vector_search(Item, "embedding", EMB_BOOK, metric="euclidean")
