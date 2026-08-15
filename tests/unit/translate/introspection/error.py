"""Error-path coverage for ``SHOW``: only the statements Milvus can
answer are wired -- anything else is an honest rejection, not an opaque
server error."""

from __future__ import annotations

import pytest

import milvusql

pytestmark = [pytest.mark.unit, pytest.mark.translate]


def test_an_unknown_show_statement_is_rejected(build_call_helper):
    with pytest.raises(milvusql.NotSupportedError, match="SHOW"):
        build_call_helper("SHOW GRANTS")
