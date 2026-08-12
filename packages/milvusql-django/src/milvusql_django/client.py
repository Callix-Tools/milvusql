"""``DatabaseClient`` -- no ``dbshell`` support. Milvus Lite has no CLI
client, and a real server's is `milvus_cli`, a separate tool this
package doesn't bundle or assume is installed."""

from __future__ import annotations

import typing as t

from django.db.backends.base.client import BaseDatabaseClient


class DatabaseClient(BaseDatabaseClient):
    executable_name = None

    @classmethod
    def settings_to_cmd_args_env(
        cls, settings_dict: dict[str, t.Any], parameters: list[str]
    ) -> tuple[list[str], dict[str, str] | None]:
        msg = (
            "milvusql-django has no dbshell -- Milvus Lite has no CLI "
            "client, and a real server's (milvus_cli) is a separate "
            "tool this package doesn't assume is installed."
        )
        raise NotImplementedError(msg)


__all__ = ["DatabaseClient"]
