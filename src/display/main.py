"""Display Agent — renders HUD to HDMI via OpenCV or logs in headless mode."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

import cv2
import httpx
import numpy as np
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse

from src.display.buffer import DisplayState, SnapshotBuffer
from src.display.schemas import HUDUpdate, HUDResponse, HealthResponse
from src.display.renderers import batch, bundle, area

logger = logging.getLogger("display")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", "1920"))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", "1080"))
DISPLAY_FPS = int(os.getenv("DISPLAY_FPS", "15"))
DISPLAY_HEADLESS = os.getenv("DISPLAY_HEADLESS", "false").lower() == "true"
APP_ID = os.getenv("APP_ID", "batch_count")
CAMERA_URL = os.getenv("CAMERA_URL", "http://localhost:8002")

_VALID_APP_IDS = {"batch_count", "bundle_check", "area_monitor"}


def _validate_config() -> None:
    if DISPLAY_WIDTH < 1:
        print(f"FATAL: DISPLAY_WIDTH={DISPLAY_WIDTH} must be > 0", file=sys.stderr)
        sys.exit(1)
    if DISPLAY_HEIGHT < 1:
        print(f"FATAL: DISPLAY_HEIGHT={DISPLAY_HEIGHT} must be > 0", file=sys.stderr)
        sys.exit(1)
    if DISPLAY_FPS < 1:
        print(f"FATAL: DISPLAY_FPS={DISPLAY_FPS} must be > 0", file=sys.stderr)
        sys.exit(1)
    if APP_ID not in _VALID_APP_IDS:
        print(f"FATAL: APP_ID={APP_ID!r} must be one of {sorted(_VALID_APP_IDS)}", file=sys.stderr)
        sys.exit(1)


_validate_config()

@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    t = threading.Thread(target=_render_loop, daemon=True, name="render-loop")
    t.start()
    logger.info("Display agent started (headless=%s)", DISPLAY_HEADLESS)
    yield


app = FastAPI(title="InventoryAI Display Agent", lifespan=_lifespan)
_state = DisplayState()
_snapshot = SnapshotBuffer()

_RENDERERS = {
    "batch_count": batch.render,
    "bundle_check": bundle.render,
    "area_monitor": area.render,
}


def _render_loop() -> None:
    canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    renderer = _RENDERERS.get(APP_ID, batch.render)
    interval = 1.0 / DISPLAY_FPS
    # _latest_frame_lock guards _latest_frame — camera frame for rendering
    _latest_frame_lock = threading.Lock()
    _latest_frame = [None]  # mutable container for closure

    # Background thread: pull frames from camera agent
    def _frame_puller() -> None:
        client = httpx.Client(timeout=2.0)
        while True:
            try:
                resp = client.get(f"{CAMERA_URL}/frame")
                if resp.status_code == 200:
                    with _latest_frame_lock:
                        _latest_frame[0] = resp.content
            except httpx.RequestError:
                pass
            time.sleep(interval)

    puller = threading.Thread(target=_frame_puller, daemon=True, name="frame-puller")
    puller.start()

    if not DISPLAY_HEADLESS:
        cv2.namedWindow("InventoryAI", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("InventoryAI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    logger.info("Render loop started (headless=%s, fps=%d, mode=%s)", DISPLAY_HEADLESS, DISPLAY_FPS, APP_ID)

    while True:
        try:
            t0 = time.monotonic()
            hud = _state.snapshot()

            # Decode latest camera frame as background
            with _latest_frame_lock:
                frame_bytes = _latest_frame[0]
            if frame_bytes:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    canvas = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                else:
                    canvas[:] = (30, 30, 30)
            else:
                canvas[:] = (30, 30, 30)

            canvas = renderer(canvas, hud)

            _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _snapshot.store(buf.tobytes())

            if not DISPLAY_HEADLESS:
                cv2.imshow("InventoryAI", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    cv2.destroyAllWindows()
                    break

            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)
        except (cv2.error, ValueError) as exc:
            logger.error("Render error: %s", exc)
            time.sleep(1.0)



@app.post("/hud", response_model=HUDResponse)
async def update_hud(body: HUDUpdate) -> HUDResponse:
    _state.update(body)
    return HUDResponse(status="ok")


@app.get("/snapshot")
async def snapshot() -> Response:
    data = _snapshot.read()
    if data is None:
        return JSONResponse({"error": "No snapshot"}, status_code=503)
    return Response(content=data, media_type="image/jpeg")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", headless=DISPLAY_HEADLESS, app_id=APP_ID)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)  # noqa: S104 — bind all interfaces for Docker
