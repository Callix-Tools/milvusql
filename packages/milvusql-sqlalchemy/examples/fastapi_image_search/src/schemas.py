"""Response models."""

from __future__ import annotations

from pydantic import BaseModel


class ImageOut(BaseModel):
    id: int
    filename: str
    caption: str


class SearchHit(BaseModel):
    id: int
    filename: str
    caption: str
    distance: float
