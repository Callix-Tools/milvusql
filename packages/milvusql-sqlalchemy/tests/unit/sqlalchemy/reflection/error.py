"""Error/edge-case coverage for ``reflection.py``'s
``columns_from_description``: a field type outside the map is an
honestly-unknown reflection, not a guess."""

from __future__ import annotations

import pytest
from milvusql_sqlalchemy.reflection import columns_from_description
from pymilvus import DataType
from sqlalchemy import types as sa_types

pytestmark = [pytest.mark.unit, pytest.mark.sqlalchemy]


class TestColumnsFromDescription:
    def test_a_field_type_outside_the_map_reflects_as_nulltype(self):
        """Phase-1 DDL never emits e.g. ``ARRAY`` -- reflecting one
        back honestly reports "unknown" rather than guessing."""
        description = {
            "fields": [
                {"name": "tags", "type": int(DataType.ARRAY), "params": {}}
            ]
        }
        columns = columns_from_description(description)
        assert isinstance(columns[0]["type"], sa_types.NullType)
