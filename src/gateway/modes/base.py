"""Abstract base class for all gateway counting modes."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMode(ABC):

    @abstractmethod
    async def on_inference_result(self, detections: list[dict]) -> None:
        """Process a new batch of detections from the inference agent."""

    @abstractmethod
    async def get_display_state(self) -> dict:
        """Return the current HUD state dict to send to the display agent."""

    @abstractmethod
    async def handle_qr(self, sku: str) -> None:
        """Handle a QR code scan containing a SKU string."""

    @abstractmethod
    def get_state(self) -> str:
        """Return the current state machine state as a string."""
