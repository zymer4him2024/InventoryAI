"""Pydantic models for display HUD updates."""

from __future__ import annotations

from typing import Dict, Optional

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
    live_count: Optional[int] = None
    target_count: Optional[int] = None
    sku: Optional[str] = None
    result: Optional[str] = None  # PASS | FAIL
    # bundle_check fields
    checklist: Optional[Dict[str, bool]] = None  # class_name → detected
    # area_monitor fields
    total_count: Optional[int] = None
    delta: Optional[int] = None
    alert: Optional[bool] = None
    location: Optional[str] = None
    last_updated: Optional[str] = None
