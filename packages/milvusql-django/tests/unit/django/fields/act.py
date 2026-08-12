"""``VectorField`` standalone (no ``Model`` attachment needed -- it
only round-trips ``list[float] <-> VECTOR(n)`` DDL text; ``milvusql``'s
DBAPI already carries the Python list through as a real bind value)."""

from __future__ import annotations

import typing as t

import pytest
from django.db.backends.base.base import BaseDatabaseWrapper
from milvusql_django.fields import VectorField

pytestmark = [pytest.mark.unit, pytest.mark.django]

#: `db_type`/`from_db_value` never read `connection` (confirmed by
#: reading `fields.py`) -- this stands in for the `BaseDatabaseWrapper`
#: their signatures declare without needing a real one built.
_NO_CONNECTION = t.cast(BaseDatabaseWrapper, None)


class TestDbType:
    def test_dim_given_renders_a_sized_vector_type(self):
        assert VectorField(dim=8).db_type(_NO_CONNECTION) == "VECTOR(8)"

    def test_no_dim_renders_the_bare_vector_type(self):
        assert VectorField().db_type(_NO_CONNECTION) == "VECTOR"


def test_get_internal_type_is_vector_field():
    assert VectorField(dim=8).get_internal_type() == "VectorField"


class TestToPython:
    def test_a_list_passes_through_unchanged(self):
        assert VectorField(dim=3).to_python([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_none_passes_through_unchanged(self):
        assert VectorField(dim=3).to_python(None) is None

    def test_a_tuple_is_converted_to_a_list(self):
        assert VectorField(dim=3).to_python((1.0, 2.0, 3.0)) == [1.0, 2.0, 3.0]


def test_from_db_value_delegates_to_to_python():
    field = VectorField(dim=2)
    assert field.from_db_value((1.0, 2.0), None, _NO_CONNECTION) == [1.0, 2.0]


def test_get_prep_value_delegates_to_to_python():
    field = VectorField(dim=2)
    assert field.get_prep_value((1.0, 2.0)) == [1.0, 2.0]


class TestDeconstruct:
    def test_a_given_dim_is_carried_in_kwargs(self):
        _, path, args, kwargs = VectorField(dim=8).deconstruct()
        assert path == "milvusql_django.fields.VectorField"
        assert args == []
        assert kwargs == {"dim": 8}

    def test_no_dim_omits_the_kwarg_entirely(self):
        _, _, _, kwargs = VectorField().deconstruct()
        assert kwargs == {}
