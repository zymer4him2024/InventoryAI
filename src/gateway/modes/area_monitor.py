"""Area Monitor mode — continuous counting with threshold alerts."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx

from src.gateway.modes.base import BaseMode
from src.gateway import config

logger = logging.getLogger("gateway.area_monitor")


class AreaMonitorMode(BaseMode):

    def __init__(self) -> None:
        self._state = "MONITORING"
        self._total_count = 0
        self._last_snapshot_count = 0
        self._last_snapshot_time = time.monotonic()
        self._alert = False

    def get_state(self) -> str:
        return self._state

    async def handle_qr(self, sku: str) -> None:
        logger.debug("QR scan ignored in area_monitor mode (sku=%s)", sku)

    async def on_inference_result(self, detections: list[dict]) -> None:
        now = time.monotonic()
        self._total_count = len(detections)

        # Check thresholds
        was_alert = self._alert
        self._alert = (
            self._total_count <= config.LOW_STOCK_THRESHOLD
            or self._total_count >= config.HIGH_STOCK_THRESHOLD
        )
        self._state = "ALERT" if self._alert else "MONITORING"

        # Write snapshot on threshold crossing
        if self._alert != was_alert:
            logger.info("Threshold crossed: count=%d state=%s", self._total_count, self._state)
            await self._write_snapshot()
            self._last_snapshot_count = self._total_count
            self._last_snapshot_time = now
            return

        # Periodic snapshot
        if (now - self._last_snapshot_time) >= config.AREA_SNAPSHOT_INTERVAL_SEC:
            await self._write_snapshot()
            self._last_snapshot_count = self._total_count
            self._last_snapshot_time = now

    async def _write_snapshot(self) -> None:
        delta = self._total_count - self._last_snapshot_count
        event = {
            "device_id": config.DEVICE_ID,
            "location_name": config.LOCATION_NAME,
            "count": self._total_count,
            "delta": delta,
            "state": self._state,
            "threshold_low": config.LOW_STOCK_THRESHOLD,
            "threshold_high": config.HIGH_STOCK_THRESHOLD,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "app_id": "area_monitor",
        }
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{config.FIREBASE_SYNC_URL}/write",
                    json={"app_id": "area_monitor", "collection": "inventory_area_snapshots", "data": event},
                    timeout=config.HEALTH_TIMEOUT,
                )
        except httpx.RequestError as exc:
            logger.error("Failed to write area snapshot: %s", exc)

    async def get_display_state(self) -> dict:
        delta = self._total_count - self._last_snapshot_count
        return {
            "app_id": "area_monitor",
            "state": self._state,
            "total_count": self._total_count,
            "delta": delta,
            "alert": self._alert,
            "location": config.LOCATION_NAME,
            "last_updated": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        }
