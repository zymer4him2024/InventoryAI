"""Pydantic models for display HUD updates."""

from __future__ import annotations


from pydantic import BaseModel


class HUDResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    headless: bool
    app_id: str


class HUDUpdate(BaseModel):
    app_id: str = ""
    state: str = ""
    # batch_count fields
    live_count: int | None = None
    target_count: int | None = None
    sku: str | None = None
    result: str | None = None  # PASS | FAIL
    # bundle_check fields
    checklist: dict[str, bool] | None = None  # class_name → detected
    # area_monitor fields
    total_count: int | None = None
    delta: int | None = None
    alert: bool | None = None
    location: str | None = None
    last_updated: str | None = None
