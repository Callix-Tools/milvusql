"""Unit coverage for ``UNION``/``INTERSECT``/``EXCEPT``.

A set operation is not a ``SELECT`` node, so it has its own dispatch
entry: each branch plans like any other query -- its reads joining the
same chain -- and the rows are combined once both frames are in hand."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]

LEFT = [{"id": 1}, {"id": 2}]
RIGHT = [{"id": 2}, {"id": 3}]


class TestPlan:
    def test_each_branch_is_its_own_read(self, build_call_helper, drive_chain):
        """Both branches read the same collection here, with different
        filters -- so they are two reads, and their results must not
        share a frame."""
        calls, _result = drive_chain(
            build_call_helper(
                "SELECT id FROM items WHERE price > 40 "
                "UNION SELECT id FROM items WHERE price < 5"
            ),
            [LEFT, RIGHT],
        )
        assert [call.kwargs["collection_name"] for call in calls] == [
            "items",
            "items",
        ]
        assert [call.kwargs["filter"] for call in calls] == [
            "price > 40",
            "price < 5",
        ]

    def test_branches_may_read_different_collections(
        self, build_call_helper, drive_chain
    ):
        calls, _result = drive_chain(
            build_call_helper(
                "SELECT id FROM items UNION SELECT id FROM cats"
            ),
            [LEFT, RIGHT],
        )
        assert [call.kwargs["collection_name"] for call in calls] == [
            "items",
            "cats",
        ]


class TestCombination:
    def test_union_removes_duplicates(self, build_call_helper, drive_chain):
        _calls, (rows, desc, rowcount, _l) = drive_chain(
            build_call_helper(
                "SELECT id FROM items UNION SELECT id FROM cats"
            ),
            [LEFT, RIGHT],
        )
        assert rows == [(1,), (2,), (3,)]
        assert [column[0] for column in desc] == ["id"]
        assert rowcount == 3

    def test_union_all_keeps_them(self, build_call_helper, drive_chain):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT id FROM items UNION ALL SELECT id FROM cats"
            ),
            [LEFT, RIGHT],
        )
        assert rows == [(1,), (2,), (2,), (3,)]

    def test_intersect_keeps_common_rows(self, build_call_helper, drive_chain):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT id FROM items INTERSECT SELECT id FROM cats"
            ),
            [LEFT, RIGHT],
        )
        assert rows == [(2,)]

    def test_except_removes_them(self, build_call_helper, drive_chain):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT id FROM items EXCEPT SELECT id FROM cats"
            ),
            [LEFT, RIGHT],
        )
        assert rows == [(1,)]

    def test_the_result_takes_the_first_branch_column_names(
        self, build_call_helper, drive_chain
    ):
        _calls, (_rows, desc, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT id AS left_id FROM items "
                "UNION SELECT id AS right_id FROM cats"
            ),
            [LEFT, RIGHT],
        )
        assert [column[0] for column in desc] == ["left_id"]

    def test_order_and_limit_apply_to_the_combined_rows(
        self, build_call_helper, drive_chain
    ):
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT id FROM items UNION SELECT id FROM cats "
                "ORDER BY 1 DESC LIMIT 2"
            ),
            [LEFT, RIGHT],
        )
        assert rows == [(3,), (2,)]

    def test_an_empty_branch_still_combines(
        self, build_call_helper, drive_chain
    ):
        """An empty read has no values to infer a dtype from; the
        concatenation has to cope with that rather than raise."""
        _calls, (rows, _d, _rc, _l) = drive_chain(
            build_call_helper(
                "SELECT id FROM items UNION SELECT id FROM cats"
            ),
            [[], RIGHT],
        )
        assert rows == [(2,), (3,)]


class TestRejected:
    def test_intersect_all_is_rejected(self, build_call_helper):
        """``INTERSECT ALL`` keeps the *minimum count* of each duplicate
        row -- a multiset operation a semi-join cannot express."""
        with pytest.raises(milvusql.NotSupportedError, match="INTERSECT ALL"):
            build_call_helper(
                "SELECT id FROM items INTERSECT ALL SELECT id FROM cats"
            )

    def test_mismatched_widths_are_rejected(
        self, build_call_helper, drive_chain
    ):
        with pytest.raises(milvusql.ProgrammingError, match="same"):
            drive_chain(
                build_call_helper(
                    "SELECT id, price FROM items UNION SELECT id FROM cats"
                ),
                [[{"id": 1, "price": 2.0}], RIGHT],
            )
