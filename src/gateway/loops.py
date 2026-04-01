"""Background loops for camera-inference pipeline and QR scanning."""

from __future__ import annotations

import asyncio
import base64
import logging
import time

import httpx

from src.gateway import config
from src.gateway.state import gw

logger = logging.getLogger("gateway")


async def inference_loop() -> None:
    """Main loop: camera -> inference -> mode -> display."""
    logger.info("Inference loop started (APP_ID=%s, interval=%.2fs)", config.APP_ID, config.INFERENCE_INTERVAL_SEC)
    while True:
        try:
            t0 = time.monotonic()

            # Fetch frame from camera
            try:
                frame_resp = await gw.client().get(
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
                inf_resp = await gw.client().post(
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
            async with gw.mode_lock:
                await gw.mode.on_inference_result(detections)

            # Update display with frame + detections
            async with gw.mode_lock:
                display_state = await gw.mode.get_display_state()
            display_state["frame_b64"] = base64.b64encode(frame_bytes).decode("ascii")
            display_state["detections"] = detections
            try:
                await gw.client().post(
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


async def qr_scan_loop() -> None:
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

            resp = await gw.client().get(f"{config.CAMERA_URL}/frame", timeout=config.HEALTH_TIMEOUT)
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
            async with gw.mode_lock:
                await gw.mode.handle_qr(qr_text)

        except (httpx.RequestError, ValueError) as exc:
            logger.error("QR scan error: %s", exc)
            await asyncio.sleep(2.0)
