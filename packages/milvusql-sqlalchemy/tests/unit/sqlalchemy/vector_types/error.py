"""Error/edge-case coverage for ``VECTOR``/``SPARSEVEC`` comparators:
a dense-only operator must not be reachable on a sparse column."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.types import SPARSEVEC
from sqlalchemy import Column, MetaData, Table

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]


class TestSparsevecComparator:
    def test_l2_distance_is_not_exposed_on_a_sparse_column(self):
        metadata = MetaData()
        items = Table("items", metadata, Column("sparse", SPARSEVEC()))
        with pytest.raises(AttributeError):
            items.c.sparse.l2_distance({0: 0.5})
