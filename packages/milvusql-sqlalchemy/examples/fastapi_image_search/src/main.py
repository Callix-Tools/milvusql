"""FastAPI image-search service: upload images with a caption, search
by a query image or by free-text (cross-modal, via CLIP's shared
image/text embedding space).

Run with:

    uv run uvicorn main:app --app-dir src --reload

See the package README for setup (bringing up Milvus via Docker
Compose) and example `curl` calls.
"""

from __future__ import annotations

import typing as t
from contextlib import asynccontextmanager

from db import bootstrap_schema, images, make_engine
from embeddings import Embedder, get_embedder
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from schemas import ImageOut, SearchHit
from sqlalchemy import delete, insert, literal_column, select
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.concurrency import run_in_threadpool

_engine: AsyncEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> t.AsyncIterator[None]:
    global _engine  # noqa: PLW0603 -- one engine for the app's lifetime, standard FastAPI lifespan pattern
    _engine = make_engine()
    await bootstrap_schema(_engine)
    yield
    await _engine.dispose()
    _engine = None


app = FastAPI(title="milvusql image search", lifespan=lifespan)


def get_engine() -> AsyncEngine:
    if _engine is None:  # pragma: no cover -- only unset outside lifespan
        msg = "database engine is not initialized"
        raise RuntimeError(msg)
    return _engine


@app.post("/images", response_model=ImageOut, status_code=201)
async def create_image(
    file: UploadFile = File(...),
    caption: str = Form(""),
    embedder: Embedder = Depends(get_embedder),
    engine: AsyncEngine = Depends(get_engine),
) -> ImageOut:
    data = await file.read()
    # CLIP inference is CPU-bound, synchronous work -- off the event
    # loop so it doesn't stall other requests.
    vector = await run_in_threadpool(embedder.embed_image, data)

    async with engine.connect() as conn:
        result = await conn.execute(
            insert(images),
            {
                "filename": file.filename or "upload",
                "caption": caption,
                "embedding": vector,
            },
        )
        await conn.commit()
        (new_id,) = result.inserted_primary_key

    return ImageOut(
        id=new_id, filename=file.filename or "upload", caption=caption
    )


@app.get("/images/{image_id}", response_model=ImageOut)
async def get_image(
    image_id: int, engine: AsyncEngine = Depends(get_engine)
) -> ImageOut:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                select(images.c.id, images.c.filename, images.c.caption).where(
                    images.c.id == image_id
                )
            )
        ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="image not found")
    return ImageOut(id=row.id, filename=row.filename, caption=row.caption)


@app.delete("/images/{image_id}", status_code=204)
async def delete_image(
    image_id: int, engine: AsyncEngine = Depends(get_engine)
) -> None:
    async with engine.connect() as conn:
        result = await conn.execute(
            delete(images).where(images.c.id == image_id)
        )
        await conn.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="image not found")


def _search_statement(vector: list[float], limit: int):
    # `literal_column("distance")` -- a synthetic column MilvusQL
    # exposes on a vector-search `SELECT` (the nearest-neighbor
    # distance itself), not one of `images`' own real columns, so it
    # can't be referenced as `images.c.distance`. See the root
    # `milvusql` package's `ast_to_pymilvus._search_rows` for where it
    # comes from.
    #
    # Rows always come back nearest-match-first regardless of the
    # index's `metric_type` -- Milvus itself does the ranking, this
    # query doesn't re-sort by the raw number. Don't read `distance`
    # as "smaller is always closer" though: for `metric_type=COSINE`
    # (this index's, and the one `<=>` matches) it's a similarity
    # score, so *larger* means more similar; for `metric_type=L2`
    # (`<->`) it's the reverse. `distance` is exposed here mainly so a
    # caller can see the raw score per hit, not to be re-sorted on.
    return (
        select(
            images.c.id,
            images.c.filename,
            images.c.caption,
            literal_column("distance"),
        )
        .order_by(images.c.embedding.cosine_distance(vector))
        .limit(limit)
    )


@app.post("/search/image", response_model=list[SearchHit])
async def search_by_image(
    file: UploadFile = File(...),
    limit: int = Query(5, ge=1, le=50),
    embedder: Embedder = Depends(get_embedder),
    engine: AsyncEngine = Depends(get_engine),
) -> list[SearchHit]:
    data = await file.read()
    vector = await run_in_threadpool(embedder.embed_image, data)
    async with engine.connect() as conn:
        rows = (await conn.execute(_search_statement(vector, limit))).all()
    return [
        SearchHit(
            id=r.id, filename=r.filename, caption=r.caption, distance=r.distance
        )
        for r in rows
    ]


@app.get("/search/text", response_model=list[SearchHit])
async def search_by_text(
    q: str,
    limit: int = Query(5, ge=1, le=50),
    embedder: Embedder = Depends(get_embedder),
    engine: AsyncEngine = Depends(get_engine),
) -> list[SearchHit]:
    # Cross-modal: CLIP's text encoder lands in the same vector space
    # as its image encoder, so a plain text query searches the exact
    # same `embedding` column/index `/search/image` does.
    vector = await run_in_threadpool(embedder.embed_text, q)
    async with engine.connect() as conn:
        rows = (await conn.execute(_search_statement(vector, limit))).all()
    return [
        SearchHit(
            id=r.id, filename=r.filename, caption=r.caption, distance=r.distance
        )
        for r in rows
    ]
