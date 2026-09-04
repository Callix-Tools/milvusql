"""What the top-k-per-group recogniser refuses to claim, and what the
translator does with the shape it does claim.

Two separate promises. The recogniser stays conservative: every shape
whose meaning is not *exactly* ``group_by_field`` + ``group_size`` is
left alone for the path that already handled it. And the statement it
does recognise is rejected by name rather than answered by reading
every vector in the collection client-side."""

from __future__ import annotations

import pytest
import sqlglot

import milvusql
from milvusql.translate._common import grouping_search

from .act import TOP_K_PER_GROUP

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def _recognize(sql: str):
    return grouping_search(sqlglot.parse_one(sql, read="milvus"))


class TestShapesNotClaimed:
    def test_rank_is_not_grouping_search(self):
        """``RANK`` keeps every row of a tie, so ``rn <= K`` can return
        more than K rows per group -- ``group_size`` cannot express
        that, so the shape is not claimed."""
        assert (
            _recognize(TOP_K_PER_GROUP.replace("ROW_NUMBER", "RANK")) is None
        )

    def test_dense_rank_is_not_either(self):
        assert (
            _recognize(TOP_K_PER_GROUP.replace("ROW_NUMBER", "DENSE_RANK"))
            is None
        )

    def test_a_scalar_order_inside_over_is_not_a_vector_search(self):
        """A real window ranking, but nothing for Grouping Search to
        do: it ranks by price, not by distance."""
        assert (
            _recognize(TOP_K_PER_GROUP.replace("embedding <=> :q", "price"))
            is None
        )

    def test_no_partition_by_means_no_groups(self):
        assert (
            _recognize(TOP_K_PER_GROUP.replace("PARTITION BY category", ""))
            is None
        )

    def test_an_outer_predicate_on_anything_else_is_not_claimed(self):
        """``rn <= 3 AND price > 10`` needs the extra conjunct pushed
        into the search's own filter, which this does not read."""
        assert _recognize(f"{TOP_K_PER_GROUP} AND price > 10") is None

    def test_an_inner_where_is_not_claimed(self):
        assert (
            _recognize(
                TOP_K_PER_GROUP.replace(
                    "FROM items", "FROM items WHERE price > 10"
                )
            )
            is None
        )


class TestRejection:
    def test_the_recognized_shape_is_refused_not_guessed_at(
        self, build_call_helper
    ):
        with pytest.raises(
            milvusql.NotSupportedError, match="Grouping Search"
        ):
            build_call_helper(TOP_K_PER_GROUP, {"q": [0.1, 0.2]})

    def test_the_message_names_the_parameters_it_would_send(
        self, build_call_helper
    ):
        """A caller reading the error should be able to write the
        equivalent ``pymilvus`` call by hand from it."""
        with pytest.raises(milvusql.NotSupportedError) as raised:
            build_call_helper(TOP_K_PER_GROUP, {"q": [0.1, 0.2]})
        message = str(raised.value)
        assert "group_by_field='category'" in message
        assert "group_size=3" in message
        assert "COSINE" in message


class TestNoSilentMistranslation:
    """The regression this shape used to be. ``FROM (SELECT ...
    ROW_NUMBER() OVER (...) AS rn FROM items) t WHERE rn <= 3`` was
    flattened to ``SELECT ... FROM items WHERE rn <= 3``: the window --
    and with it the whole vector search inside its ``OVER`` -- was
    discarded, and Milvus was asked for a field literally named ``rn``.
    Wrong rows, no error."""

    def test_a_computed_column_is_never_flattened_away(
        self, build_call_helper
    ):
        with pytest.raises(milvusql.NotSupportedError):
            build_call_helper(TOP_K_PER_GROUP, {"q": [0.1, 0.2]})

    def test_a_window_the_recognizer_passes_over_still_keeps_its_columns(
        self, build_call_helper
    ):
        """``RANK`` is not claimed above -- but it must not be flattened
        into a phantom ``rn`` field either: it goes to the relational
        engine, which fetches the columns the window actually reads."""
        call = build_call_helper(
            "SELECT id, rn FROM ("
            "  SELECT id, RANK() OVER (PARTITION BY category ORDER BY price)"
            "  AS rn FROM items"
            ") t WHERE rn <= 3"
        )
        assert "rn" not in call.kwargs["output_fields"]
        assert sorted(call.kwargs["output_fields"]) == [
            "category",
            "id",
            "price",
        ]

    def test_a_genuinely_trivial_subquery_still_flattens(
        self, build_call_helper
    ):
        """The guard is on *computed* projections only: SQLAlchemy's
        ``Query.count()`` wraps every query in a plain relabelling
        subquery, and that must keep collapsing into one RPC."""
        call = build_call_helper(
            "SELECT id FROM (SELECT id, name FROM items WHERE id > 5) t "
            "WHERE id < 100"
        )
        assert call.method == "query"
        assert call.kwargs["collection_name"] == "items"
        assert "id > 5" in call.kwargs["filter"]
        assert "id < 100" in call.kwargs["filter"]
