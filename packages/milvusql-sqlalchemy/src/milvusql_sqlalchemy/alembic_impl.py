"""Alembic support: registers ``milvusql``'s own ``DefaultImpl``
subclass so ``alembic.runtime.migration.MigrationContext.configure()``
doesn't raise a bare ``KeyError`` looking this dialect up by name --
confirmed directly, ``alembic.ddl.impl.DefaultImpl.get_by_dialect``
does an unconditional ``_impls[dialect.name]``, with no
"unregistered dialect falls back to the generic default" behavior the
way, say, a missing entry in a plain ``dict.get(..., default)`` would
suggest. *Importing* this module is what registers it:
``DefaultImpl``'s own metaclass (``ImplMeta.__new__``) populates that
``_impls`` dict the moment a subclass with ``__dialect__`` set is
defined -- no separate registration call, no Alembic-side entry-point
mechanism to hook into. ``dialect.py`` imports this unconditionally at
its own import time, guarded by an ``ImportError`` check so this
package doesn't gain a hard dependency on Alembic just to be usable
without it (``alembic`` isn't in this package's own
``dependencies`` -- only in the workspace's test group, since only the
Alembic integration suite needs it installed)."""

from __future__ import annotations

from alembic.ddl.impl import DefaultImpl


class MilvusImpl(DefaultImpl):
    __dialect__ = "milvusql"
    #: Milvus has no multi-statement rollback (D7 -- the same
    #: reasoning ``MilvusDialect.do_rollback``'s own module docstring
    #: gives): wrapping a migration in a transaction Alembic can never
    #: actually roll back would silently misrepresent a guarantee this
    #: backend can't keep the moment anything failed mid-migration.
    #: ``alembic.ddl.impl.DefaultImpl`` itself already defaults this
    #: to ``False`` (confirmed directly), so this assignment changes
    #: nothing behaviorally -- it's here so the reason is documented
    #: at the one place a reader would look for it on this dialect,
    #: not left to a coincidence of Alembic's own base-class default.
    transactional_ddl = False


__all__ = ["MilvusImpl"]
