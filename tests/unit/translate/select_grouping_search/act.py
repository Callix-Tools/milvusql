"""Recognising top-k-per-group -- Milvus's *Grouping Search*.

``ROW_NUMBER() OVER (PARTITION BY category ORDER BY embedding <=> :q)``
filtered on ``rn <= K`` is SQL's spelling of "the best K entities per
category", which Milvus answers natively with ``search(...,
group_by_field='category', group_size=K)``. These pin what the
recogniser claims and -- just as importantly -- what it refuses to
claim, since a shape it describes wrongly produces wrong rows rather
than an error."""

from __future__ import annotations

import pytest
import sqlglot

from milvusql.translate._common import grouping_search

pytestmark = [pytest.mark.unit, pytest.mark.translate]

TOP_K_PER_GROUP = (
    "SELECT id, category, rn FROM ("
    "  SELECT id, category, ROW_NUMBER() OVER ("
    "    PARTITION BY category ORDER BY embedding <=> :q"
    "  ) AS rn FROM items"
    ") t WHERE rn <= 3"
)


def _recognize(sql: str):
    return grouping_search(sqlglot.parse_one(sql, read="milvus"))


class TestRecognizedShape:
    def test_every_search_parameter_is_read_off_the_statement(self):
        found = _recognize(TOP_K_PER_GROUP)
        assert found is not None
        assert found.table_name == "items"
        assert found.group_by_field == "category"
        assert found.group_size == 3
        assert found.anns_field == "embedding"
        assert found.metric_type == "COSINE"
        assert found.rank_alias == "rn"

    def test_the_metric_follows_the_distance_operator(self):
        """The grammar's own operator table decides ``metric_type``;
        the recogniser does not hard-code cosine."""
        found = _recognize(TOP_K_PER_GROUP.replace("<=>", "<->"))
        assert found is not None
        assert found.metric_type == "L2"

    def test_a_strict_rank_bound_keeps_one_row_fewer(self):
        """``rn < 4`` is three rows per group, not four."""
        found = _recognize(TOP_K_PER_GROUP.replace("rn <= 3", "rn < 4"))
        assert found is not None
        assert found.group_size == 3

    def test_the_alias_is_whatever_the_statement_called_it(self):
        renamed = TOP_K_PER_GROUP.replace("rn", "position")
        found = _recognize(renamed)
        assert found is not None
        assert found.rank_alias == "position"
        assert found.group_size == 3
