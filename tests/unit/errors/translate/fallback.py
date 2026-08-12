"""D8's catch-all: any exception type ``errors.translate()`` has no
specific mapping for falls back to a plain ``DatabaseError``."""

from __future__ import annotations

import pytest

import milvusql
from milvusql.dbapi import errors

pytestmark = [pytest.mark.unit, pytest.mark.errors]


def test_unrecognized_exception_type_falls_back_to_database_error():
    assert isinstance(
        errors.translate(ValueError("huh")), milvusql.DatabaseError
    )
