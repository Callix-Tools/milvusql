"""Interactive CLI chat with the catalog agent.

    uv run src/main.py

Requires the catalog to already exist -- run `uv run src/seed.py`
once first. Unlike the earlier version of this example, this doesn't
need to re-run schema bootstrap here too: that workaround was for
Milvus Lite, whose "loaded" state lives in the process that loaded it
rather than on disk. Against a real server (this example's target,
via `docker-compose.yaml`), `seed.py`'s own `LOAD` -- or, since D2
revised, even just its own first query -- is enough; a fresh process
querying the same server sees the same load state, and `search_products`
would auto-load it again anyway if it somehow didn't.
"""

from __future__ import annotations

import asyncio

from agent import CatalogDeps, catalog_agent
from db import make_engine
from embeddings import get_embedder


async def main() -> None:
    engine = make_engine()
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
