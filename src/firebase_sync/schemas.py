"""Pydantic models for Firebase sync events and SKU config."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SKUConfig(BaseModel):
    sku: str
    part_class: str = ""
    target_count: int = 0
    required_classes: list[str] = Field(default_factory=list)
    customer_id: str = ""
    tolerance: int = 0


class BatchEvent(BaseModel):
    device_id: str
    sku: str
    result: str  # PASS | FAIL
    detected_count: int
    target_count: int
    timestamp: str
    app_id: str = "batch_count"


class BundleEvent(BaseModel):
    device_id: str
    sku: str
    result: str  # PASS | FAIL
    required_classes: list[str]
    detected_classes: list[str]
    missing_classes: list[str]
    timestamp: str
    app_id: str = "bundle_check"


class AreaSnapshot(BaseModel):
    device_id: str
    location_name: str
    count: int
    delta: int
    state: str  # MONITORING | ALERT
    threshold_low: int
    threshold_high: int
    timestamp: str
    app_id: str = "area_monitor"


class WriteRequest(BaseModel):
    app_id: str
    collection: str
    data: dict


class WriteResponse(BaseModel):
    status: str
    collection: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    simulation: bool
