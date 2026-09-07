"""Error-path coverage for vector-search ``SELECT`` dispatch."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_order_by_a_non_column_scalar_expression_raises(build_call_helper):
    """A plain scalar ``ORDER BY`` is sorted client-side (see
    ``select_order_by``), but only by a bare column -- an expression
    (``ORDER BY id + 1``) has no natural Milvus field to fetch and sort
    by."""
    with pytest.raises(milvusql.NotSupportedError, match="ORDER BY"):
        build_call_helper("SELECT id FROM items ORDER BY id + 1 LIMIT 5")


class TestDistanceScoreInTheSelectListIsRejected:
    """A search returns its own score; a projection cannot compute one.
    Asking for ``embedding <=> :q`` in the SELECT list used to send
    ``output_fields=['id', '']`` and hand back ``None`` for every
    distance -- indistinguishable from a row that genuinely has none."""

    def test_a_distance_operator_in_the_select_list_raises(
        self, build_call_helper
    ):
        with pytest.raises(
            milvusql.NotSupportedError, match="embedding <=> :q"
        ):
            build_call_helper(
                "SELECT id, embedding <=> :q AS dist FROM items LIMIT 5",
                {"q": [0.1] * 8},
            )

    def test_it_raises_even_alongside_the_matching_order_by(
        self, build_call_helper
    ):
        """The search shape itself is supported -- only the projected
        copy of the scoring expression is not, and it must not slip
        through just because the statement is otherwise a valid
        search."""
        with pytest.raises(milvusql.NotSupportedError, match="ORDER BY"):
            build_call_helper(
                "SELECT id, embedding <=> :q AS dist FROM items "
                "ORDER BY embedding <=> :q LIMIT 5",
                {"q": [0.1] * 8},
            )

    def test_a_bm25_score_in_the_select_list_raises(self, build_call_helper):
        """``BM25_SCORE`` is the full-text ranking -- a search score by
        the same argument, so it takes the same route."""
        with pytest.raises(milvusql.NotSupportedError, match="BM25_SCORE"):
            build_call_helper(
                "SELECT id, BM25_SCORE(body, :q) AS s FROM items LIMIT 5",
                {"q": "vector databases"},
            )

    def test_the_spelling_the_message_recommends_really_works(
        self, build_call_helper
    ):
        """The rejection points at ``ORDER BY <distance>`` plus a
        ``distance`` column; that has to compile to a real search which
        actually returns the score, or the message is sending callers
        somewhere broken."""
        call = build_call_helper(
            "SELECT id, distance FROM items ORDER BY embedding <=> :q LIMIT 5",
            {"q": [0.1] * 8},
        )
        assert call.method == "search"
        assert call.kwargs["output_fields"] == ["id", "distance"]
        rows, _description, count, _ = call.postprocess(
            [[{"id": 7, "distance": 0.25, "entity": {}}]]
        )
        assert rows == [(7, 0.25)]
        assert count == 1
