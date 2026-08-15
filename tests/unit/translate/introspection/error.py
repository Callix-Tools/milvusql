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


def test_a_show_qualifier_is_rejected_not_silently_dropped(
    build_call_helper,
):
    """`SHOW TABLES LIKE 'x%'` used to match on the keyword alone and
    return the full unfiltered listing -- a plausible wrong answer."""
    with pytest.raises(milvusql.NotSupportedError, match="qualifier"):
        build_call_helper("SHOW TABLES LIKE 'item%'")
