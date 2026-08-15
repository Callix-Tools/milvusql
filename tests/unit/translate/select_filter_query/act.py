"""Unit coverage for ``translate.ast_to_pymilvus.build_call`` on a
plain (non-vector) ``SELECT``, dispatched to ``query``."""

from __future__ import annotations

import pytest

import milvusql

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


class TestTrivialSubqueryIsFlattened:
    """The shape SQLAlchemy's legacy ``Query.count()``/``Query.exists()``
    wrap *every* query in: a subquery that does nothing but relabel
    columns and (optionally) filter a single real table. Rejecting it
    outright would break that common, harmless idiom, so
    ``_flatten_trivial_subquery`` unwraps it instead -- see
    ``_build_select``."""

    def test_a_subquery_with_no_where_flattens_to_the_real_table(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT id FROM (SELECT items.id AS id FROM items) AS anon_1"
        )
        assert call.kwargs["collection_name"] == "items"
        assert "filter" not in call.kwargs

    def test_the_inner_where_survives_flattening(self, build_call_helper):
        """``Query(Model).filter(...).count()`` puts the filter
        *inside* the subquery, not the outer query -- confirmed
        directly against SQLAlchemy's own compiled SQL."""
        call = build_call_helper(
            "SELECT id FROM (SELECT items.id AS id FROM items "
            "WHERE items.category = :cat) AS anon_1",
            {"cat": "book"},
        )
        assert call.kwargs["filter"] == 'category == "book"'

    def test_inner_and_outer_where_are_anded_together(self, build_call_helper):
        call = build_call_helper(
            "SELECT id FROM (SELECT items.id AS id FROM items "
            "WHERE items.category = :cat) AS anon_1 WHERE id > :n",
            {"cat": "book", "n": 1},
        )
        assert call.kwargs["filter"] == '(id > 1 and category == "book")'


class TestClausesWithNoMilvusEquivalent:
    """``query()`` takes a filter and a row cap, nothing else -- so a
    statement carrying one of these has to be evaluated client-side
    rather than sent as-is and quietly dropped."""

    def test_distinct_on_one_collection_deduplicates(self, build_call_helper):
        call = build_call_helper("SELECT DISTINCT cid FROM items")
        rows, _desc, _rc, _l = call.postprocess([{"cid": 1}, {"cid": 1}])
        assert rows == [(1,)]

    def test_offset_on_one_collection_is_applied(self, build_call_helper):
        call = build_call_helper("SELECT id FROM items LIMIT 2 OFFSET 1")
        rows, _desc, _rc, _l = call.postprocess(
            [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
        )
        assert rows == [(2,), (3,)]


class TestJsonAndArrayFilters:
    def test_a_json_path_comparison_renders_with_brackets(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT id FROM t WHERE meta['brand'] = :b LIMIT 5", {"b": "acme"}
        )
        assert call.kwargs["filter"] == 'meta["brand"] == "acme"'

    def test_a_nested_json_path_renders_every_step(self, build_call_helper):
        call = build_call_helper(
            "SELECT id FROM t WHERE meta['spec']['w'] > 10 LIMIT 5"
        )
        assert call.kwargs["filter"] == 'meta["spec"]["w"] > 10'

    def test_array_contains_and_friends_pass_through_by_name(
        self, build_call_helper
    ):
        call = build_call_helper(
            "SELECT id FROM t WHERE ARRAY_CONTAINS(tags, :v) LIMIT 5",
            {"v": "x"},
        )
        assert call.kwargs["filter"] == 'array_contains(tags, "x")'
        call = build_call_helper(
            "SELECT id FROM t WHERE ARRAY_CONTAINS_ANY(tags, :vals) LIMIT 5",
            {"vals": ["a", "b"]},
        )
        assert call.kwargs["filter"] == 'array_contains_any(tags, ["a", "b"])'

    def test_an_unknown_filter_function_is_rejected_with_the_roster(
        self, build_call_helper
    ):
        with pytest.raises(milvusql.NotSupportedError, match="ARRAY_CONTAINS"):
            build_call_helper(
                "SELECT id FROM t WHERE MYSTERY_FN(name) = :v LIMIT 5",
                {"v": "x"},
            )


class TestBareBooleanPredicates:
    """Django compiles ``.filter(active=True)`` to the bare column
    (``WHERE "active"``), and Milvus rejects a bare field reference as
    a predicate -- confirmed against a real server: "predicate is not a
    boolean expression". Boolean positions get the explicit
    comparison; value positions stay untouched."""

    def test_a_bare_boolean_column_becomes_an_explicit_comparison(
        self, build_call_helper
    ):
        call = build_call_helper("SELECT id FROM t WHERE active LIMIT 5")
        assert call.kwargs["filter"] == "active == true"

    def test_inside_and_or_not(self, build_call_helper):
        call = build_call_helper(
            "SELECT id FROM t WHERE active AND (deleted OR id > 5) LIMIT 5"
        )
        assert call.kwargs["filter"] == (
            "(active == true and ((deleted == true or id > 5)))"
        )
        call = build_call_helper("SELECT id FROM t WHERE NOT active LIMIT 5")
        assert call.kwargs["filter"] == "not (active == true)"

    def test_value_positions_are_not_wrapped(self, build_call_helper):
        call = build_call_helper(
            "SELECT id FROM t WHERE flag = other_flag LIMIT 5"
        )
        assert call.kwargs["filter"] == "flag == other_flag"

    def test_array_length_parses_to_its_own_node_and_still_renders(
        self, build_call_helper
    ):
        """``ARRAY_LENGTH(x)`` parses to ``exp.ArraySize``, not an
        ``Anonymous`` call -- caught by a real CI failure against a
        real server."""
        call = build_call_helper(
            "SELECT id FROM t WHERE ARRAY_LENGTH(nums) = :n LIMIT 5",
            {"n": 3},
        )
        assert call.kwargs["filter"] == "array_length(nums) == 3"
