"""Pydantic models for inference request/response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str = Field(..., min_length=1, max_length=64)
    score: float = Field(..., ge=0.0, le=1.0)
    box: list[float] = Field(..., min_length=4, max_length=4, description="[x_min, y_min, width, height]")


class InferenceResponse(BaseModel):
    success: bool = True
    inference_ms: float = Field(default=0.0, ge=0.0)
    detections: list[Detection] = Field(default_factory=list, max_length=200)


class HealthResponse(BaseModel):
    status: str
    simulation: bool
    model: str
