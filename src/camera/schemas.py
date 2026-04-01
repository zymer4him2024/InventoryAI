"""Pydantic models for camera agent endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    camera_ok: bool
    simulate: bool
