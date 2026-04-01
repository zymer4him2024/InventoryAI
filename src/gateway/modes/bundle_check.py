"""Bundle Check mode — detect whether a required set of part types are all present."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx

from src.gateway.modes.base import BaseMode
from src.gateway import config

logger = logging.getLogger("gateway.bundle_check")


class BundleCheckMode(BaseMode):

    def __init__(self) -> None:
        self._state = "IDLE"
        self._sku = ""
        self._required_classes: list[str] = []
        self._detected_classes: set[str] = set()
        self._result: str | None = None
        self._scan_start: float | None = None
        self._result_time: float | None = None

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

        required = data.get("required_classes", [])
        if not required:
            logger.warning("SKU %s has no required_classes", sku)
            return

        self._sku = sku
        self._required_classes = required
        self._detected_classes = set()
        self._result = None
        self._scan_start = time.monotonic()
        self._state = "SCANNING"
        logger.info("Bundle scan started: SKU=%s required=%s", sku, required)

    async def on_inference_result(self, detections: list[dict]) -> None:
        now = time.monotonic()

        # Auto-reset after result hold
        if self._state in ("PASS", "FAIL") and self._result_time is not None:
            if now - self._result_time >= config.RESULT_HOLD_SEC:
                self._state = "IDLE"
                self._result = None
                self._sku = ""
                self._required_classes = []
                self._detected_classes = set()
                logger.info("Auto-reset to IDLE")
            return

        if self._state != "SCANNING":
            return

        # Accumulate detected classes
        for det in detections:
            label = det.get("label", "")
            if label in self._required_classes:
                self._detected_classes.add(label)

        # Check if all required classes found
        if self._detected_classes >= set(self._required_classes):
            self._state = "PASS"
            self._result = "PASS"
            self._result_time = now
            logger.info("Bundle PASS: all %d classes detected", len(self._required_classes))
            await self._write_event()
            return

        # Check timeout
        if self._scan_start and (now - self._scan_start) >= config.BUNDLE_TIMEOUT_SEC:
            self._state = "FAIL"
            self._result = "FAIL"
            self._result_time = now
            missing = set(self._required_classes) - self._detected_classes
            logger.info("Bundle FAIL: missing %s", missing)
            await self._write_event()

    async def _write_event(self) -> None:
        missing = sorted(set(self._required_classes) - self._detected_classes)
        event = {
            "device_id": config.DEVICE_ID,
            "sku": self._sku,
            "result": self._result,
            "required_classes": self._required_classes,
            "detected_classes": sorted(self._detected_classes),
            "missing_classes": missing,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "app_id": "bundle_check",
        }
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{config.FIREBASE_SYNC_URL}/write",
                    json={"app_id": "bundle_check", "collection": "inventory_bundle_events", "data": event},
                    timeout=config.HEALTH_TIMEOUT,
                )
        except httpx.RequestError as exc:
            logger.error("Failed to write bundle event: %s", exc)

    async def get_display_state(self) -> dict:
        checklist = {cls: cls in self._detected_classes for cls in self._required_classes}
        return {
            "app_id": "bundle_check",
            "state": self._state,
            "sku": self._sku,
            "checklist": checklist,
            "result": self._result,
        }
