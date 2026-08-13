"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on a
plain (non-vector) ``SELECT``, dispatched to ``query``."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.translate]


class TestSelectFilterQuery:
    def test_plain_select_dispatches_to_query(self, build_call_helper):
        call = build_call_helper("SELECT id, category FROM items LIMIT 10")
        assert call.method == "query"
        assert call.kwargs == {
            "collection_name": "items",
            "output_fields": ["id", "category"],
            "limit": 10,
        }

    def test_where_and_or_not_and_parens_render_correctly(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT id FROM items WHERE category = :cat "
            "AND (id > :n OR id = :m) LIMIT 5",
            {"cat": "book", "n": 1, "m": 2},
        )
        assert call.kwargs["filter"] == (
            '(category == "book" and ((id > 1 or id == 2)))'
        )

    def test_not_wraps_its_operand(self, build_call_helper):
        call = build_call_helper(
            "SELECT id FROM items WHERE NOT (category = :cat) LIMIT 5",
            {"cat": "book"},
        )
        assert call.kwargs["filter"] == 'not ((category == "book"))'

    def test_in_renders_as_milvus_native_in_list_syntax(
        self, build_call_helper
    ):
        call = build_call_helper(
            'SELECT id FROM items WHERE "id" IN (:a, :b, :c) LIMIT 5',
            {"a": 1, "b": 2, "c": 3},
        )
        assert call.kwargs["filter"] == "id in [1, 2, 3]"

    def test_not_in_renders_via_the_shared_not_wrapper(
        self, build_call_helper
    ):
        call = build_call_helper(
            'SELECT id FROM items WHERE "id" NOT IN (:a, :b) LIMIT 5',
            {"a": 1, "b": 2},
        )
        assert call.kwargs["filter"] == "not (id in [1, 2])"

    def test_an_empty_in_list_matches_nothing(self, build_call_helper):
        call = build_call_helper(
            'SELECT id FROM items WHERE "id" IN () LIMIT 5'
        )
        assert call.kwargs["filter"] == "false"

    def test_like_renders_as_milvus_native_like_syntax(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT id FROM items WHERE category LIKE :pat LIMIT 5",
            {"pat": "book%"},
        )
        assert call.kwargs["filter"] == 'category like "book%"'

    def test_not_like_renders_via_the_negate_flag_not_a_not_wrapper(
        self, build_call_helper
    ):
        """``NOT LIKE`` compiles to a ``negate`` flag on the same
        ``exp.Like`` node (unlike ``NOT IN``/``IS NOT NULL``, which
        wrap in a separate ``exp.Not``)."""
        call = build_call_helper(
            "SELECT id FROM items WHERE category NOT LIKE :pat LIMIT 5",
            {"pat": "book%"},
        )
        assert call.kwargs["filter"] == 'not (category like "book%")'

    def test_is_null_renders_as_milvus_native_is_null_syntax(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT id FROM items WHERE category IS NULL LIMIT 5"
        )
        assert call.kwargs["filter"] == "category is null"

    def test_is_not_null_renders_via_the_shared_not_wrapper(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT id FROM items WHERE category IS NOT NULL LIMIT 5"
        )
        assert call.kwargs["filter"] == "not (category is null)"

    def test_between_transpiles_to_a_gte_lte_pair(self, build_call_helper):
        """Milvus's filter DSL has no ``BETWEEN`` keyword at all
        (confirmed directly: a bare ``id BETWEEN 1 AND 2`` is a syntax
        error), so it transpiles to the equivalent range check."""
        call = build_call_helper(
            "SELECT id FROM items WHERE id BETWEEN :lo AND :hi LIMIT 5",
            {"lo": 1, "hi": 10},
        )
        assert call.kwargs["filter"] == "(id >= 1 and id <= 10)"

    def test_not_between_renders_via_the_shared_not_wrapper(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT id FROM items WHERE id NOT BETWEEN :lo AND :hi LIMIT 5",
            {"lo": 1, "hi": 10},
        )
        assert call.kwargs["filter"] == "not ((id >= 1 and id <= 10))"

    def test_no_limit_clause_falls_back_to_milvus_own_ceiling(
        self, build_call_helper
    ):
        """A bare ``Model.objects.all()`` compiles to no ``LIMIT`` at
        all (confirmed against ``DatabaseOperations.no_limit_value()``
        returning ``None``) -- this used to silently cap at an
        arbitrary ``10``, now it falls back to Milvus's own per-call
        ceiling instead."""
        call = build_call_helper("SELECT id FROM items")
        assert call.kwargs["limit"] == 16384

    def test_limit_resolves_a_bound_parameter_not_just_a_literal(
        self, build_call_helper
    ):
        """SQLAlchemy-generated text binds ``LIMIT`` by default
        (confirmed directly: ``.limit(5)`` compiles to ``LIMIT
        :param_1``) rather than inlining a literal."""
        call = build_call_helper("SELECT id FROM items LIMIT :lim", {"lim": 7})
        assert call.kwargs["limit"] == 7

    def test_query_postprocess_shapes_rows_and_description(
        self, build_call_helper
    ):
        call = build_call_helper("SELECT id, category FROM items LIMIT 10")
        rows, description, rowcount, lastrowid = call.postprocess(
            [{"id": 1, "category": "book"}]
        )
        assert rows == [(1, "book")]
        assert description == [
            ("id", None, None, None, None, None, True),
            ("category", None, None, None, None, None, True),
        ]
        assert rowcount == 1
        assert lastrowid is None
