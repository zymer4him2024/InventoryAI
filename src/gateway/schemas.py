"""Pydantic models for gateway endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class JobRequest(BaseModel):
    sku: str = Field(..., min_length=1, max_length=128)


class JobResponse(BaseModel):
    status: str
    sku: str
    state: str


class StatusResponse(BaseModel):
    app_id: str
    device_id: str
    state: str
    display: dict


class HealthResponse(BaseModel):
    status: str
    app_id: str
    device_id: str
