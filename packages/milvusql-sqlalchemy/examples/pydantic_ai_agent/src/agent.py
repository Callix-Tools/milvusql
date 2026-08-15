"""A `pydantic-ai` shopping-assistant agent backed by the Milvus
catalog: `search_products` runs a real vector search through
`milvusql-sqlalchemy`, `get_product` looks one up by id. The model
never sees raw SQL -- only these two typed tools and their results.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import AGENT_MODEL
from db import products
from embeddings import Embedder
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from sqlalchemy import literal_column, select
from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass
class CatalogDeps:
    """Runtime dependencies the tools below need -- injected once per
    `agent.run()` call via `deps=`, not imported as globals, so tests
    can swap in a throwaway engine/embedder (see the package's own
    tests for the analogous pattern in `milvusql-sqlalchemy` itself)."""

    engine: AsyncEngine
    embedder: Embedder


class ProductResult(BaseModel):
    id: int
    name: str
    description: str
    category: str
    price: float
    #: The raw vector-search score for this hit, `None` for a plain
    #: `get_product` lookup. Higher means more similar for this
    #: index's `metric_type=COSINE` -- see `search_products`'s
    #: docstring.
    relevance: float | None = None


catalog_agent = Agent(
    AGENT_MODEL,
    deps_type=CatalogDeps,
    output_type=str,
    # Defers resolving `AGENT_MODEL` against its provider (which needs
    # a real API key) until the agent actually runs -- so this module
    # can be imported, and tested with
    # `catalog_agent.override(model=TestModel())`, without one set.
    defer_model_check=True,
    system_prompt=(
        "You are a helpful shopping assistant for an outdoor-gear "
        "catalog. Use search_products to find items matching what "
        "the customer describes, and get_product to look up a "
        "specific item by id. Always ground your answers in tool "
        "results -- never invent a product, price, or id that a tool "
        "didn't actually return. If no product is a good match, say "
        "so instead of guessing."
    ),
)


@catalog_agent.tool
async def search_products(
    ctx: RunContext[CatalogDeps], query: str, limit: int = 5
) -> list[ProductResult]:
    """Search the catalog for products semantically similar to
    `query` -- a natural-language description of what the customer
    wants (e.g. "something warm for cold hikes"), not a keyword
    filter. Returns the `limit` closest matches, best match first."""
    vector = ctx.deps.embedder.embed_text(query)
    stmt = (
        select(
            products.c.id,
            products.c.name,
            products.c.description,
            products.c.category,
            products.c.price,
            # A synthetic column MilvusQL exposes on a vector-search
            # `SELECT` (the hit's own score), not one of `products`'
            # real columns -- see the root `milvusql` package's
            # `ast_to_pymilvus._search_rows` for where it comes from.
            literal_column("distance"),
        )
        .order_by(products.c.embedding.cosine_distance(vector))
        .limit(limit)
    )
    async with ctx.deps.engine.connect() as conn:
        rows = (await conn.execute(stmt)).all()
    return [
        ProductResult(
            id=r.id,
            name=r.name,
            description=r.description,
            category=r.category,
            price=r.price,
            relevance=r.distance,
        )
        for r in rows
    ]


@catalog_agent.tool
async def get_product(
    ctx: RunContext[CatalogDeps], product_id: int
) -> ProductResult | None:
    """Look up one product by its exact id (e.g. after
    `search_products` returned it and the customer asks for more
    detail). Returns `None` if no product has that id."""
    stmt = select(
        products.c.id,
        products.c.name,
        products.c.description,
        products.c.category,
        products.c.price,
    ).where(products.c.id == product_id)
    async with ctx.deps.engine.connect() as conn:
        row = (await conn.execute(stmt)).first()
    if row is None:
        return None
    return ProductResult(
        id=row.id,
        name=row.name,
        description=row.description,
        category=row.category,
        price=row.price,
    )
