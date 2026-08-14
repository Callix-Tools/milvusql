"""Interactive CLI chat with the catalog agent.

    python -m catalog.main

Requires the catalog to already exist -- run `python -m catalog.seed`
once first.
"""

from __future__ import annotations

import asyncio

from catalog.agent import CatalogDeps, catalog_agent
from catalog.db import bootstrap_schema, make_async_engine
from catalog.embeddings import get_embedder


async def main() -> None:
    # Re-runs the same guarded steps `seed.py` already ran -- cheap
    # when nothing's missing, and necessary for one reason beyond
    # that: Milvus Lite's "loaded" state lives in the *process* that
    # loaded it, not in the on-disk file, so a fresh process (this
    # one) reopening the same file needs its own `load_collection()`
    # call even though `seed.py` already made one in its own process.
    # A real (non-Lite) server keeps load state independently of any
    # one client connection, so this is a no-op there -- but it's
    # correct either way, which is why `search_products` doesn't have
    # to special-case Lite at all.
    await asyncio.to_thread(bootstrap_schema)

    engine = make_async_engine()
    deps = CatalogDeps(engine=engine, embedder=get_embedder())
    message_history = None

    print("Catalog assistant -- ask about products ('quit' to exit).")
    try:
        while True:
            try:
                user_input = input("> ").strip()
            except EOFError:
                break
            if not user_input or user_input.lower() in {"quit", "exit"}:
                break

            result = await catalog_agent.run(
                user_input, deps=deps, message_history=message_history
            )
            print(result.output)
            # Carried into the next turn so the agent remembers
            # earlier context (e.g. "tell me more about the second
            # one").
            message_history = result.all_messages()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
