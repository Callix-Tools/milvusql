"""What the Milvus on the other end of a connection can actually do.

Every "does the server do this, or do we?" decision in this package is
currently a constant: ``GROUP BY`` always goes to the Polars engine in
:mod:`translate.relational`, ``TEXT`` is always an analyzer-enabled
``VARCHAR``, the row ceiling is always
:data:`translate._common.DEFAULT_QUERY_LIMIT`. That is right for the
one server line this release supports and wrong as a *rule* -- Milvus
3.0 evaluates single-collection ``ORDER BY``/``GROUP BY``/aggregates in
the kernel, carries ``TEXT`` as a real field type and drops the fixed
per-call ceiling, so the same SQL should resolve into server-side
parameters there and fall back to the client-side path only where it
has to.

This module is where that question gets asked once, from the two
version strings that answer it -- the server's own
``get_server_version()`` and the installed ``pymilvus`` -- instead of
being re-decided ad hoc at each call site. Nothing consults it yet:
the delegating paths land one at a time behind the flags, and a flag
landing before its consumer is deliberate, so each of those changes is
a small diff against a fixed, tested capability table rather than a
version check invented on the spot.

A flag is ``True`` only where that is *known*, never where it is
merely likely: the failure mode of guessing high is a silently wrong
answer from a server that quietly ignores a parameter it does not
understand, while guessing low costs only the client-side path this
release already takes.

Pure: no I/O, no ``pymilvus`` client. :meth:`Connection.capabilities`
(and its async twin) do the one RPC and hand the string here.
"""

from __future__ import annotations

import importlib.metadata as _metadata
import re
from dataclasses import dataclass

#: Milvus's per-call row ceiling on the 2.x line, repeated from
#: ``translate._common`` rather than imported: that module is the
#: translator's constant and this one is a *property of a server*, and
#: importing it here would make the pure capability table depend on the
#: translator it is meant to inform.
ROW_CEILING_2X = 16384

#: What Milvus Lite reports as its version -- ``"milvus_lite-3.2.0"``
#: (confirmed directly against an embedded ``.db`` connection). The
#: trailing number is the ``milvus-lite`` *package* version and has
#: nothing to do with the Milvus server release line, so it must never
#: be parsed as one: reading "3.2.0" as "Milvus 3.2" would turn on
#: every 3.0 server-side flag for the one deployment that supports the
#: fewest of them.
_LITE_PREFIX = "milvus_lite"

#: ``v2.6.4``, ``2.6.4``, ``v2.6.4-gpu``: a real server reports its
#: build tag, so only the leading ``major.minor`` is dependable.
_RELEASE_RE = re.compile(r"v?(\d+)\.(\d+)")

#: The first release line this project supports at all (``pymilvus>=2.6``
#: in ``pyproject.toml``, Milvus 2.6.x in the README's compatibility
#: table). Used as the floor for flags whose feature predates it: this
#: is deliberately *not* a claim about which release added the feature,
#: only about the oldest pairing that is tested.
_SUPPORTED_FLOOR = (2, 6)

#: Where the engine takes over the single-collection relational surface.
_KERNEL_SQL = (3, 0)

Release = tuple[int, int]


@dataclass(frozen=True)
class ServerCapabilities:
    """One connection's answer to "who evaluates what".

    The flags describe *the server and client in play*, not what this
    release of ``milvusql`` currently delegates to them -- a ``True``
    here is permission to build a server-side call, not a promise that
    some code path already does.
    """

    #: Verbatim ``get_server_version()``, ``None`` when unknown.
    server_version: str | None
    #: ``(major, minor)``, or ``None`` for Milvus Lite and for anything
    #: that did not parse -- an unreadable version is never guessed at.
    server_release: Release | None
    #: ``(major, minor)`` of the installed ``pymilvus``.
    client_release: Release | None
    #: Milvus Lite, running in this process rather than over gRPC.
    embedded: bool

    #: ``search(group_by_field=..., group_size=...)`` -- top-k per
    #: group in one RPC (the discussion's "Grouping Search", the
    #: mapping for ``ROW_NUMBER() OVER (PARTITION BY ...)``).
    grouping_search: bool
    #: Single-collection ``GROUP BY`` with ``COUNT``/``SUM``/``AVG``/
    #: ``MIN``/``MAX`` computed in the kernel, addressed through
    #: ``query(group_by_fields=...)``.
    server_side_group_by: bool
    #: Single-collection ``ORDER BY`` sorted per segment and
    #: merge-sorted across query nodes instead of in the client.
    server_side_order_by: bool
    #: ``TEXT`` as a real field type rather than an analyzer-enabled
    #: ``VARCHAR``.
    native_text_field: bool
    #: Server-issued search cursor (``SearchIteratorV2Info``).
    search_iterator_cursor: bool
    #: Server-issued query cursor (``QueryIteratorCursor``), the
    #: replacement for paging by primary key.
    query_iterator_cursor: bool
    #: Iterator reads come back primary-key-ordered, which is what
    #: makes the primary-key page loop in ``translate._common`` safe.
    #: ``False`` for Milvus Lite, which ignores the ``iterator`` flag.
    ordered_iterator_pages: bool
    #: Rows one ``query``/``search`` RPC can return, ``None`` where the
    #: server no longer imposes a fixed one.
    per_call_row_ceiling: int | None


def parse_release(version: str | None) -> Release | None:
    """``(major, minor)`` out of a reported version, or ``None`` when
    there is nothing dependable to read.

    Milvus Lite is ``None`` on purpose and not an oversight -- see
    :data:`_LITE_PREFIX`.
    """
    if not version or version.startswith(_LITE_PREFIX):
        return None
    match = _RELEASE_RE.match(version.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def installed_client_release() -> Release | None:
    """``(major, minor)`` of the ``pymilvus`` actually importable here.

    Read from package metadata for the same reason ``milvusql``'s own
    ``__version__`` is: it is the one number the installer wrote.
    """
    try:
        return parse_release(_metadata.version("pymilvus"))
    except _metadata.PackageNotFoundError:  # pragma: no cover
        return None


def capabilities_for(
    server_version: str | None,
    client_release: Release | None = None,
) -> ServerCapabilities:
    """The capability table for one server/client pairing.

    ``client_release`` defaults to the installed ``pymilvus``. It is a
    parameter at all because half of these are client-gated as much as
    server-gated: ``group_by_fields`` does not appear anywhere in
    pymilvus 2.6.17 (checked against the installed package), so a 3.0
    server reached through a 2.6 client still cannot be asked for a
    kernel ``GROUP BY``.
    """
    if client_release is None:
        client_release = installed_client_release()
    release = parse_release(server_version)
    embedded = bool(server_version and server_version.startswith(_LITE_PREFIX))

    def both_at(floor: Release) -> bool:
        """Server and client both at ``floor`` or newer. Neither half
        is assumed when its version did not parse."""
        return (
            release is not None
            and release >= floor
            and client_release is not None
            and client_release >= floor
        )

    # Everything the 3.0 engine absorbs is gated on the same pairing:
    # the server has to evaluate it and the client has to be able to
    # ask. `search_iterator_cursor` is server-gated but client-floored
    # lower -- `SearchIteratorV2` ships in pymilvus 2.6.17 already
    # (checked against the installed package, which falls back with a
    # warning when the server has no cursor to hand out), so a `False`
    # here means "do not assume one", not "refuse to try".
    kernel_sql = both_at(_KERNEL_SQL)
    return ServerCapabilities(
        server_version=server_version,
        server_release=release,
        client_release=client_release,
        embedded=embedded,
        # Not `both_at(2, 4)`, where `group_by_field` predates this
        # project: below the supported floor nothing here is tested, so
        # the client-side path is the honest answer. Milvus Lite is
        # `False` for the stronger reason that grouping search against
        # the embedded engine is unverified -- guessing `True` there
        # would send a parameter that may be ignored rather than
        # refused, and an ignored `group_by_field` returns plain top-k
        # silently.
        grouping_search=not embedded and both_at(_SUPPORTED_FLOOR),
        server_side_group_by=kernel_sql,
        server_side_order_by=kernel_sql,
        native_text_field=kernel_sql,
        search_iterator_cursor=(
            release is not None
            and release >= _KERNEL_SQL
            and client_release is not None
            and client_release >= _SUPPORTED_FLOOR
        ),
        query_iterator_cursor=kernel_sql,
        # An unknown server is assumed to order iterator pages, which
        # is what the page loop in `translate._common` already assumes
        # -- and it re-checks every page at runtime regardless, so this
        # flag informs planning, it does not replace that guard.
        ordered_iterator_pages=not embedded,
        per_call_row_ceiling=(
            None
            if release is not None and release >= _KERNEL_SQL
            else ROW_CEILING_2X
        ),
    )


__all__ = [
    "ROW_CEILING_2X",
    "Release",
    "ServerCapabilities",
    "capabilities_for",
    "installed_client_release",
    "parse_release",
]
