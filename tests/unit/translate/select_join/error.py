"""Error-path coverage for ``translate.relational``: the shapes it must
refuse rather than guess at.

The theme is the same one the single-call translator already follows --
a wrong answer that looks right is worse than a clean error. A truncated
read, an unresolvable column name and a projection that cannot be
requested from Milvus all raise here instead of producing plausible
rows."""

from __future__ import annotations

import pytest

import milvusql
from milvusql.translate._common import DEFAULT_QUERY_LIMIT

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestAmbiguousColumns:
    def test_a_bare_column_across_two_collections_is_rejected(
        self, build_call_helper
    ):
        """Milvus is asked for output fields *before* any row comes
        back, and no collection schema is available at translate time,
        so there is nothing to resolve a bare ``id`` against -- and
        attaching it to the wrong collection would silently return
        someone else's column."""
        with pytest.raises(milvusql.ProgrammingError, match="ambiguous"):
            build_call_helper(
                "SELECT id FROM items AS i JOIN cats AS c ON i.cat_id = c.id"
            )

    def test_an_unknown_qualifier_is_rejected(self, build_call_helper):
        with pytest.raises(milvusql.ProgrammingError, match="unknown table"):
            build_call_helper(
                "SELECT x.id FROM items AS i JOIN cats AS c ON i.cat_id = c.id"
            )

    def test_a_bare_column_on_one_collection_still_works(
        self, build_call_helper
    ):
        """The ambiguity is what is rejected, not the spelling: with a
        single collection in scope a bare name resolves to it."""
        call = build_call_helper(
            "SELECT category, COUNT(*) FROM items GROUP BY category"
        )
        assert call.kwargs["output_fields"] == ["category"]


class TestUnsupportedShapes:
    def test_select_star_across_a_join_is_rejected(self, build_call_helper):
        """``*`` cannot be turned into an ``output_fields`` list per
        collection without a schema, and asking each side for ``*``
        would fetch both collections whole, vectors included."""
        with pytest.raises(milvusql.NotSupportedError, match="SELECT \\*"):
            build_call_helper(
                "SELECT * FROM items AS i JOIN cats AS c ON i.cat_id = c.id"
            )

    def test_join_using_is_rejected(self, build_call_helper):
        with pytest.raises(milvusql.NotSupportedError, match="USING"):
            build_call_helper(
                "SELECT i.id FROM items AS i JOIN cats AS c USING (cat_id)"
            )

    def test_an_unaliased_subquery_in_from_is_rejected(
        self, build_call_helper
    ):
        with pytest.raises(milvusql.NotSupportedError, match="aliased"):
            build_call_helper(
                "SELECT cat_id, COUNT(*) FROM "
                "(SELECT cat_id FROM items) GROUP BY cat_id"
            )

    def test_a_cte_is_rejected(self, build_call_helper):
        """``WITH`` parses, but nothing here plans it yet -- saying so
        beats silently reading the wrong collection."""
        with pytest.raises(milvusql.NotSupportedError, match="WITH"):
            build_call_helper(
                "WITH hot AS (SELECT id FROM items) "
                "SELECT h.id FROM hot AS h JOIN cats AS c ON h.id = c.id"
            )

    def test_a_vector_search_without_a_limit_is_rejected(
        self, build_call_helper
    ):
        with pytest.raises(milvusql.NotSupportedError, match="needs a LIMIT"):
            build_call_helper(
                "SELECT i.id FROM items AS i JOIN cats AS c "
                "ON i.cat_id = c.id ORDER BY i.embedding <=> :q",
                {"q": [0.1, 0.2]},
            )


class TestRowCeilingIsDetected:
    """A join or a grouped aggregate computed over *some* of the
    matching rows is a wrong answer with nothing to show it is wrong --
    the same reasoning, and the same ceiling, as the single-collection
    aggregate and ``ORDER BY`` paths."""

    def test_a_grouped_read_at_the_ceiling_raises(self, build_call_helper):
        call = build_call_helper(
            "SELECT category, COUNT(*) FROM items GROUP BY category"
        )
        at_ceiling = [{"category": "book"}] * DEFAULT_QUERY_LIMIT
        with pytest.raises(milvusql.NotSupportedError, match="row ceiling"):
            call.postprocess(at_ceiling)

    def test_the_message_names_the_collection(self, build_call_helper):
        call = build_call_helper(
            "SELECT category, COUNT(*) FROM items GROUP BY category"
        )
        with pytest.raises(milvusql.NotSupportedError, match="'items'"):
            call.postprocess([{"category": "book"}] * DEFAULT_QUERY_LIMIT)

    def test_a_joined_read_at_the_ceiling_raises(
        self, build_call_helper, drive_chain
    ):
        with pytest.raises(milvusql.NotSupportedError, match="row ceiling"):
            drive_chain(
                build_call_helper(
                    "SELECT i.id, c.title FROM items AS i "
                    "JOIN cats AS c ON i.cat_id = c.id"
                ),
                [
                    [{"id": 1, "cat_id": 10}] * DEFAULT_QUERY_LIMIT,
                    [{"id": 10, "title": "books"}],
                ],
            )

    def test_a_bounded_vector_search_is_not_subject_to_it(
        self, build_call_helper
    ):
        """An ANN search returns exactly the top-k it was asked for, so
        a full result is the expected outcome, not evidence of
        truncation."""
        call = build_call_helper(
            "SELECT i.id, c.title FROM items AS i JOIN cats AS c "
            "ON i.cat_id = c.id ORDER BY i.embedding <=> :q LIMIT 2",
            {"q": [0.1, 0.2]},
        )
        hits = [
            [
                {"id": 1, "distance": 0.1, "entity": {"cat_id": 10}},
                {"id": 2, "distance": 0.2, "entity": {"cat_id": 10}},
            ]
        ]
        assert call.then(hits).kwargs["collection_name"] == "cats"
