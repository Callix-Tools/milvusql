"""Unit coverage for ``vector_search``/``hybrid_search``'s own input
validation. Both reach for a live connection (``connections[using]``)
only after validating every arm's ``metric`` -- ``vector_search``
checks before touching ``model`` at all, ``hybrid_search`` checks
after reading ``model._meta`` but still before any I/O -- so the
``ValueError`` branch needs neither Django bootstrap nor a Milvus
instance. The real, executed round trip (and ``hybrid_search``'s
actual ``NotSupportedError`` behavior) is integration-covered in
``tests/integration/django/search``."""

from __future__ import annotations

import typing as t

import pytest
from milvusql_django.expressions import hybrid_search, vector_search

pytestmark = [pytest.mark.unit, pytest.mark.django]

#: `vector_search` never reads `model` before raising on a bad metric
#: (confirmed by reading `expressions.py`) -- stands in for the
#: `type[t.Any]` its signature declares without building a real model.
_UNUSED_MODEL = t.cast("type[t.Any]", None)


def test_vector_search_rejects_an_unknown_metric_before_touching_the_model():
    with pytest.raises(ValueError, match="unknown metric 'euclidean'"):
        vector_search(
            _UNUSED_MODEL, "embedding", [0.1] * 8, metric="euclidean"
        )


class _Field:
    def __init__(self, column):
        self.column = column


class _Meta:
    db_table = "items"
    local_fields = (_Field("id"), _Field("category"), _Field("embedding"))


class _Model:
    _meta = _Meta


def test_hybrid_search_rejects_an_unknown_metric_in_any_arm():
    with pytest.raises(ValueError, match="unknown metric 'euclidean'"):
        hybrid_search(
            _Model, [("embedding", "euclidean", [0.1] * 8, 1.0)], k=5
        )
