"""Batch Count mode — count N instances of a single part class, compare to target."""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime, timezone

import httpx

from src.gateway.modes.base import BaseMode
from src.gateway import config

logger = logging.getLogger("gateway.batch_count")


class BatchCountMode(BaseMode):

    def __init__(self) -> None:
        self._state = "IDLE"
        self._sku = ""
        self._part_class = ""
        self._target_count = 0
        self._tolerance = 0
        self._live_count = 0
        self._result: str | None = None
        self._counting_start: float | None = None
        self._result_time: float | None = None
        self._counts_buffer: list[int] = []

    def get_state(self) -> str:
        return self._state

    async def handle_qr(self, sku: str) -> None:
        if self._state != "IDLE":
            return
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{config.FIREBASE_SYNC_URL}/load_sku",
                    params={"sku": sku},
                    timeout=config.HEALTH_TIMEOUT,
                )
                if resp.status_code != 200:
                    logger.warning("SKU %s not found: %s", sku, resp.text)
                    return
                data = resp.json()
        except (httpx.RequestError, KeyError, ValueError) as exc:
            logger.error("Failed to load SKU %s: %s", sku, exc)
            return

        self._sku = sku
        self._part_class = data.get("part_class", "")
        self._target_count = data.get("target_count", 0)
        self._tolerance = data.get("tolerance", 0)
        self._live_count = 0
        self._result = None
        self._counts_buffer = []
        self._counting_start = time.monotonic()
        self._state = "COUNTING"
        logger.info("Job started: SKU=%s class=%s target=%d", sku, self._part_class, self._target_count)

    async def on_inference_result(self, detections: list[dict]) -> None:
        now = time.monotonic()

        # Auto-reset after result hold
        if self._state in ("PASS", "FAIL") and self._result_time is not None:
            if now - self._result_time >= config.RESULT_HOLD_SEC:
                self._state = "IDLE"
                self._result = None
                self._sku = ""
                logger.info("Auto-reset to IDLE")
            return

        if self._state != "COUNTING":
            return

        # Count detections matching part_class
        counts = Counter(d.get("label", "") for d in detections)
        detected = counts.get(self._part_class, 0) if self._part_class else sum(counts.values())
        self._live_count = detected
        self._counts_buffer.append(detected)

        # Check if counting window has elapsed
        if self._counting_start and (now - self._counting_start) >= config.COUNTING_WINDOW_SEC:
            avg_count = round(sum(self._counts_buffer) / len(self._counts_buffer)) if self._counts_buffer else 0
            self._live_count = avg_count

            if abs(avg_count - self._target_count) <= self._tolerance:
                self._state = "PASS"
                self._result = "PASS"
            else:
                self._state = "FAIL"
                self._result = "FAIL"

            self._result_time = now
            logger.info("Result: %s (detected=%d target=%d)", self._result, avg_count, self._target_count)

            # Fire event to firebase
            await self._write_event(avg_count)

    async def _write_event(self, detected_count: int) -> None:
        event = {
            "device_id": config.DEVICE_ID,
            "sku": self._sku,
            "result": self._result,
            "detected_count": detected_count,
            "target_count": self._target_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "app_id": "batch_count",
        }
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{config.FIREBASE_SYNC_URL}/write",
                    json={"app_id": "batch_count", "collection": "inventory_batch_events", "data": event},
                    timeout=config.HEALTH_TIMEOUT,
                )
        except httpx.RequestError as exc:
            logger.error("Failed to write batch event: %s", exc)

    async def get_display_state(self) -> dict:
        return {
            "app_id": "batch_count",
            "state": self._state,
            "live_count": self._live_count,
            "target_count": self._target_count,
            "sku": self._sku,
            "result": self._result,
        }
