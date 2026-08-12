"""``DatabaseClient`` has no ``dbshell`` -- unit coverage for the one
thing it does: refuse loudly rather than pretend a CLI client exists."""

from __future__ import annotations

import pytest
from milvusql_django.client import DatabaseClient

pytestmark = [pytest.mark.unit, pytest.mark.django]


def test_settings_to_cmd_args_env_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="no dbshell"):
        DatabaseClient.settings_to_cmd_args_env({}, [])
