"""Camera Agent — captures frames from USB camera or generates test images."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse

from src.camera.schemas import HealthResponse

logger = logging.getLogger("camera")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1920"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "1080"))
SIMULATE_CAMERA = os.getenv("SIMULATE_CAMERA", "false").lower() == "true"


def _validate_config() -> None:
    if CAMERA_INDEX < 0:
        print(f"FATAL: CAMERA_INDEX={CAMERA_INDEX} must be >= 0", file=sys.stderr)
        sys.exit(1)
    if CAMERA_WIDTH < 1:
        print(f"FATAL: CAMERA_WIDTH={CAMERA_WIDTH} must be > 0", file=sys.stderr)
        sys.exit(1)
    if CAMERA_HEIGHT < 1:
        print(f"FATAL: CAMERA_HEIGHT={CAMERA_HEIGHT} must be > 0", file=sys.stderr)
        sys.exit(1)


_validate_config()


@dataclass
class CameraState:
    """Holds camera capture state. All fields guarded by lock."""

    latest_frame: Optional[bytes] = None
    camera_ok: bool = False
    # Invariant: lock guards latest_frame and camera_ok.
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = CameraState()


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    t = threading.Thread(target=_capture_loop, daemon=True, name="camera-capture")
    t.start()
    logger.info("Camera agent started (simulate=%s)", SIMULATE_CAMERA)
    yield


app = FastAPI(title="InventoryAI Camera Agent", lifespan=_lifespan)


def _capture_loop() -> None:
    if SIMULATE_CAMERA:
        logger.info("Simulation mode — generating test frames")
        with _state.lock:
            _state.camera_ok = True
        while True:
            img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            img[:] = (40, 40, 40)
            cv2.putText(img, "SIMULATED FRAME", (CAMERA_WIDTH // 2 - 250, CAMERA_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            ts = time.strftime("%H:%M:%S")
            cv2.putText(img, ts, (CAMERA_WIDTH // 2 - 80, CAMERA_HEIGHT // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with _state.lock:
                _state.latest_frame = buf.tobytes()
            time.sleep(1.0 / 10)
        return

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        logger.error("Cannot open camera %d", CAMERA_INDEX)
        return

    with _state.lock:
        _state.camera_ok = True
    logger.info("Camera %d opened (%dx%d)", CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            with _state.lock:
                _state.camera_ok = False
            time.sleep(0.5)
            continue
        with _state.lock:
            _state.camera_ok = True
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with _state.lock:
            _state.latest_frame = buf.tobytes()



@app.get("/frame")
async def get_frame() -> Response:
    with _state.lock:
        frame = _state.latest_frame
    if frame is None:
        return JSONResponse({"error": "No frame available"}, status_code=503)
    return Response(content=frame, media_type="image/jpeg")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    with _state.lock:
        camera_ok = _state.camera_ok
    return HealthResponse(status="ok", camera_ok=camera_ok, simulate=SIMULATE_CAMERA)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)  # noqa: S104 — bind all interfaces for Docker
