"""Django backend bootstrap: ``settings.configure()``/``django.setup()``
can only run once per process, and several test modules import
``dj_app.models.Item`` at module level (a real ``Model``, whose
metaclass needs the app registry populated the moment the class body
runs) -- pytest imports every test module during collection, before
any fixture executes, so this has to happen as top-level code here,
not inside a fixture function: ``conftest.py`` is always imported
before sibling/descendant test modules, fixtures are not.
Each test gets a real Milvus server via ``connection`` (a
``testcontainers``-backed instance shared per pytest-xdist worker, with
per-test collection cleanup -- see
``tests/fixtures/containers/milvus_server.py``), so ``pytest-randomly``
can freely reorder tests without collection-name collisions. ``dj_app``
is a real importable app package
(``tests/dj_app/``) -- ``django.test.utils.isolate_apps()`` doesn't
work for a throwaway label with no backing package, it tries to
import it as a real module and raises ``ModuleNotFoundError``.

``tests/`` now has an ``__init__.py`` of its own (required so pytest's
rootless import mode can tell apart the many same-named ``act.py``/
``error.py`` files under ``unit``/``integration``), so pytest inserts
``tests/``'s *parent* onto ``sys.path`` rather than ``tests/`` itself
-- ``dj_app`` would no longer resolve as a bare top-level import.
Every other module still spells it ``from dj_app.models import
Item``, unchanged, so this puts ``tests/`` back on ``sys.path``
directly instead of rewriting every one of those imports."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import django
from django.conf import settings

settings.configure(
    DATABASES={"default": {"ENGINE": "milvusql_django", "NAME": ""}},
    INSTALLED_APPS=["dj_app"],
    USE_TZ=True,
    # Django's plain `AutoField` (its own implicit default for a model
    # with no explicit primary key, `dj_app.models.Item` included)
    # assumes a 32-bit range (`connection.ops.integer_field_range`,
    # confirmed directly: `(-2147483648, 2147483647)`) -- and silently
    # short-circuits any `.filter(pk=X)`/`.get(pk=X)` whose value
    # falls outside it to an empty result via `IntegerFieldOverflow`
    # raising `EmptyResultSet`, *without ever issuing a query*. A real
    # Milvus server's actual `auto_id` allocator hands out large,
    # snowflake-style ids (confirmed directly: values like
    # 468353823765898985, routinely ~4x10^17) that are nowhere near
    # 32-bit range -- Milvus Lite's small sequential ids never
    # triggered this. `BigAutoField`'s range is the full 64 bits
    # Milvus's own ids actually live in.
    DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
)
django.setup()

pytest_plugins = ["tests.fixtures"]
