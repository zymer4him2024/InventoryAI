"""Inference Agent — Hailo-8 inference or mock simulation."""

from __future__ import annotations

import logging
import os
import random
import sys
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile

from src.inference.schemas import Detection, InferenceResponse, HealthResponse

logger = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

HEF_PATH = os.getenv("HEF_PATH", "")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.35"))
MOCK_LABELS = [s.strip() for s in os.getenv("MOCK_LABELS", "bolt_m6,washer_m6,nut_m6").split(",") if s.strip()]
MOCK_MIN_COUNT = int(os.getenv("MOCK_MIN_COUNT", "1"))
MOCK_MAX_COUNT = int(os.getenv("MOCK_MAX_COUNT", "5"))


def _validate_config() -> None:
    if not (0.0 <= CONF_THRESHOLD <= 1.0):
        print(f"FATAL: CONF_THRESHOLD={CONF_THRESHOLD} must be in [0.0, 1.0]", file=sys.stderr)
        sys.exit(1)
    if MOCK_MIN_COUNT < 1:
        print(f"FATAL: MOCK_MIN_COUNT={MOCK_MIN_COUNT} must be >= 1", file=sys.stderr)
        sys.exit(1)
    if MOCK_MAX_COUNT < 1:
        print(f"FATAL: MOCK_MAX_COUNT={MOCK_MAX_COUNT} must be >= 1", file=sys.stderr)
        sys.exit(1)
    if MOCK_MIN_COUNT > MOCK_MAX_COUNT:
        print(f"FATAL: MOCK_MIN_COUNT={MOCK_MIN_COUNT} > MOCK_MAX_COUNT={MOCK_MAX_COUNT}", file=sys.stderr)
        sys.exit(1)


_validate_config()

_simulation_mode = True
_hailo_runner = None

app = FastAPI(title="InventoryAI Inference Agent")


def _try_load_hailo() -> bool:
    global _hailo_runner, _simulation_mode
    if not HEF_PATH or not os.path.isfile(HEF_PATH):
        logger.info("No HEF model at %r — simulation mode", HEF_PATH)
        return False
    try:
        from hailo_platform import HEF, VDevice, ConfigureParams, FormatType  # noqa: F401
        hef = HEF(HEF_PATH)
        vdevice = VDevice()
        configure_params = ConfigureParams.create_from_hef(hef, interface=FormatType.HW_ONLY)
        network_group = vdevice.configure(hef, configure_params)[0]
        _hailo_runner = {
            "hef": hef,
            "vdevice": vdevice,
            "network_group": network_group,
            "input_vstreams": network_group.input_vstreams,
            "output_vstreams": network_group.output_vstreams,
        }
        _simulation_mode = False
        logger.info("Hailo-8 model loaded: %s", HEF_PATH)
        return True
    except (ImportError, OSError) as exc:
        logger.warning("Hailo init failed (%s) — simulation mode", exc)
        return False


def _mock_inference(image_bytes: bytes) -> InferenceResponse:
    t0 = time.perf_counter()
    time.sleep(random.uniform(0.02, 0.06))
    detections = []
    for label in MOCK_LABELS:
        count = random.randint(MOCK_MIN_COUNT, MOCK_MAX_COUNT)
        for _ in range(count):
            x = random.randint(50, 1600)
            y = random.randint(50, 900)
            w = random.randint(30, 120)
            h = random.randint(30, 120)
            detections.append(Detection(
                label=label,
                score=round(random.uniform(0.5, 0.99), 2),
                box=[float(x), float(y), float(w), float(h)],
            ))
    elapsed = (time.perf_counter() - t0) * 1000
    return InferenceResponse(success=True, inference_ms=round(elapsed, 2), detections=detections)


def _hailo_inference(image_bytes: bytes) -> InferenceResponse:
    t0 = time.perf_counter()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return InferenceResponse(success=False, inference_ms=0.0, detections=[])

    runner = _hailo_runner
    if runner is None:
        return InferenceResponse(success=False, inference_ms=0.0, detections=[])

    input_vstream = runner["input_vstreams"][0]
    h, w = input_vstream.shape[0], input_vstream.shape[1]
    resized = cv2.resize(frame, (w, h))
    input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

    with runner["network_group"].activate():
        raw_output = runner["network_group"].run([input_data])

    detections = []
    for output in raw_output:
        for det in output:
            score = float(det[4]) if len(det) > 4 else 0.0
            if score < CONF_THRESHOLD:
                continue
            class_id = int(det[5]) if len(det) > 5 else 0
            detections.append(Detection(
                label=str(class_id),
                score=round(score, 3),
                box=[float(det[0]), float(det[1]), float(det[2] - det[0]), float(det[3] - det[1])],
            ))

    elapsed = (time.perf_counter() - t0) * 1000
    return InferenceResponse(success=True, inference_ms=round(elapsed, 2), detections=detections[:200])


@app.on_event("startup")
async def _startup() -> None:
    _try_load_hailo()
    logger.info("Inference agent started (simulation=%s, labels=%s)", _simulation_mode, MOCK_LABELS)


@app.post("/inference", response_model=InferenceResponse)
async def inference(image: UploadFile = File(...)) -> InferenceResponse:
    image_bytes = await image.read()
    if _simulation_mode:
        return _mock_inference(image_bytes)
    return _hailo_inference(image_bytes)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", simulation=_simulation_mode, model=HEF_PATH or "mock")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104 — bind all interfaces for Docker
