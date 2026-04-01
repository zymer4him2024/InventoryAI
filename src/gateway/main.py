"""Gateway Agent — state machine orchestrator for all InventoryAI modes."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Type
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from src.gateway import config
from src.gateway.schemas import JobRequest, JobResponse, StatusResponse, HealthResponse
from src.gateway.modes.base import BaseMode
from src.gateway.modes.batch_count import BatchCountMode
from src.gateway.modes.bundle_check import BundleCheckMode
from src.gateway.modes.area_monitor import AreaMonitorMode

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

config.validate()

_MODE_MAP: Dict[str, Type[BaseMode]] = {
    "batch_count": BatchCountMode,
    "bundle_check": BundleCheckMode,
    "area_monitor": AreaMonitorMode,
}


@dataclass
class GatewayState:
    """Holds gateway runtime state: active mode and shared HTTP client."""

    mode: BaseMode = field(default_factory=lambda: _MODE_MAP[config.APP_ID]())
    http_client: Optional[httpx.AsyncClient] = None
    # Invariant: mode_lock guards all mode method calls (on_inference_result,
    # get_display_state, handle_qr, get_state).
    mode_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def client(self) -> httpx.AsyncClient:
        if self.http_client is None or self.http_client.is_closed:
            self.http_client = httpx.AsyncClient()
        return self.http_client


_gw = GatewayState()


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    asyncio.create_task(_inference_loop())
    asyncio.create_task(_qr_scan_loop())
    logger.info("Gateway started — APP_ID=%s DEVICE_ID=%s", config.APP_ID, config.DEVICE_ID)
    yield
    if _gw.http_client and not _gw.http_client.is_closed:
        await _gw.http_client.aclose()


app = FastAPI(title=f"InventoryAI Gateway ({config.APP_ID})", lifespan=_lifespan)


async def _inference_loop() -> None:
    """Main loop: camera -> inference -> mode -> display."""
    logger.info("Inference loop started (APP_ID=%s, interval=%.2fs)", config.APP_ID, config.INFERENCE_INTERVAL_SEC)
    while True:
        try:
            t0 = time.monotonic()

            # Fetch frame from camera
            try:
                frame_resp = await _gw.client().get(
                    f"{config.CAMERA_URL}/frame",
                    timeout=config.HEALTH_TIMEOUT,
                )
                if frame_resp.status_code != 200:
                    await asyncio.sleep(config.INFERENCE_INTERVAL_SEC)
                    continue
                frame_bytes = frame_resp.content
            except httpx.RequestError:
                await asyncio.sleep(config.INFERENCE_INTERVAL_SEC)
                continue

            # Send to inference
            try:
                inf_resp = await _gw.client().post(
                    f"{config.INFERENCE_URL}/inference",
                    files={"image": ("frame.jpg", frame_bytes, "image/jpeg")},
                    timeout=10.0,
                )
                if inf_resp.status_code != 200:
                    await asyncio.sleep(config.INFERENCE_INTERVAL_SEC)
                    continue
                inf_data = inf_resp.json()
            except httpx.RequestError:
                await asyncio.sleep(config.INFERENCE_INTERVAL_SEC)
                continue

            detections = inf_data.get("detections", [])

            # Process through mode
            async with _gw.mode_lock:
                await _gw.mode.on_inference_result(detections)

            # Update display
            async with _gw.mode_lock:
                display_state = await _gw.mode.get_display_state()
            try:
                await _gw.client().post(
                    f"{config.DISPLAY_URL}/hud",
                    json=display_state,
                    timeout=config.HEALTH_TIMEOUT,
                )
            except httpx.RequestError:
                pass

            elapsed = time.monotonic() - t0
            remaining = config.INFERENCE_INTERVAL_SEC - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)

        except (httpx.RequestError, ValueError) as exc:
            logger.error("Inference loop error: %s", exc)
            await asyncio.sleep(1.0)


async def _qr_scan_loop() -> None:
    """Background loop: poll camera for QR codes."""
    try:
        from pyzbar.pyzbar import decode as qr_decode
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("pyzbar not available — QR scanning disabled")
        return

    logger.info("QR scan loop started (interval=%.1fs)", config.QR_SCAN_INTERVAL_SEC)
    last_qr = ""
    last_qr_time = 0.0

    while True:
        try:
            await asyncio.sleep(config.QR_SCAN_INTERVAL_SEC)

            resp = await _gw.client().get(f"{config.CAMERA_URL}/frame", timeout=config.HEALTH_TIMEOUT)
            if resp.status_code != 200:
                continue

            nparr = np.frombuffer(resp.content, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue

            codes = qr_decode(frame)
            if not codes:
                continue

            qr_text = codes[0].data.decode("utf-8", errors="ignore").strip()
            now = time.monotonic()

            # Debounce: same QR within 10s
            if qr_text == last_qr and (now - last_qr_time) < 10.0:
                continue

            last_qr = qr_text
            last_qr_time = now
            logger.info("QR scanned: %s", qr_text)
            async with _gw.mode_lock:
                await _gw.mode.handle_qr(qr_text)

        except (httpx.RequestError, ValueError) as exc:
            logger.error("QR scan error: %s", exc)
            await asyncio.sleep(2.0)



@app.post("/job", response_model=JobResponse)
async def create_job(req: JobRequest) -> JobResponse:
    async with _gw.mode_lock:
        await _gw.mode.handle_qr(req.sku)
        state = _gw.mode.get_state()
    return JobResponse(status="ok", sku=req.sku, state=state)


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    async with _gw.mode_lock:
        display_state = await _gw.mode.get_display_state()
        state = _gw.mode.get_state()
    return StatusResponse(
        app_id=config.APP_ID,
        device_id=config.DEVICE_ID,
        state=state,
        display=display_state,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", app_id=config.APP_ID, device_id=config.DEVICE_ID)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104 — bind all interfaces for Docker
