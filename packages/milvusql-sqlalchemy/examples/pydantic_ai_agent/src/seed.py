"""One-time bootstrap + seed data for the `products` collection.

    uv run src/seed.py
"""

from __future__ import annotations

import asyncio

from db import bootstrap_schema, make_engine, products
from embeddings import get_embedder
from sqlalchemy import insert

SAMPLE_PRODUCTS = [
    {
        "name": "Trailblazer 2 Hiking Boots",
        "description": "Waterproof leather hiking boots with ankle "
        "support, for rocky trails and cold weather.",
        "category": "footwear",
        "price": 129.99,
    },
    {
        "name": "AeroFlex Running Shoes",
        "description": "Lightweight breathable running shoes with "
        "responsive foam cushioning for road running.",
        "category": "footwear",
        "price": 89.99,
    },
    {
        "name": "Summit 40L Backpack",
        "description": "A durable 40-liter backpacking pack with a "
        "padded hip belt, built for multi-day hikes.",
        "category": "outdoor gear",
        "price": 159.00,
    },
    {
        "name": "Everyday Commuter Backpack",
        "description": "A slim 20-liter backpack with a padded laptop "
        "sleeve, for cycling or walking to the office.",
        "category": "bags",
        "price": 54.50,
    },
    {
        "name": "StormShell Rain Jacket",
        "description": "A packable waterproof rain jacket with sealed "
        "seams, for hiking and everyday wet-weather use.",
        "category": "outerwear",
        "price": 99.00,
    },
    {
        "name": "ThermaCore Fleece Jacket",
        "description": "A warm midweight fleece jacket for cold-weather "
        "layering on winter hikes.",
        "category": "outerwear",
        "price": 74.00,
    },
    {
        "name": "TrailBrew Camping Coffee Maker",
        "description": "A compact pour-over coffee maker for making "
        "coffee at a campsite or on a backpacking trip.",
        "category": "camping gear",
        "price": 24.99,
    },
    {
        "name": "PackLite 2-Person Tent",
        "description": "An ultralight 2-person backpacking tent that "
        "sets up in under five minutes.",
        "category": "camping gear",
        "price": 219.00,
    },
]


async def main() -> None:
    engine = make_engine()
    try:
        await bootstrap_schema(engine)
        embedder = get_embedder()

        async with engine.connect() as conn:
            await conn.execute(
                insert(products),
                [
                    {
                        **item,
                        "embedding": embedder.embed_text(
                            f"{item['name']}: {item['description']}"
                        ),
                    }
                    for item in SAMPLE_PRODUCTS
                ],
            )
            await conn.commit()
    finally:
        await engine.dispose()

    print(f"seeded {len(SAMPLE_PRODUCTS)} products")


if __name__ == "__main__":
    asyncio.run(main())
