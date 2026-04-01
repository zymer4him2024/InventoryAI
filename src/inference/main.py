"""Inference Agent — Hailo-8 inference or mock simulation."""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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


@dataclass
class InferenceState:
    """Holds Hailo runtime state. Set once at startup, read-only thereafter."""

    simulation: bool = True
    hailo_runner: Optional[Dict[str, Any]] = None


_state = InferenceState()


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    _try_load_hailo()
    logger.info("Inference agent started (simulation=%s, labels=%s)", _state.simulation, MOCK_LABELS)
    yield


app = FastAPI(title="InventoryAI Inference Agent", lifespan=_lifespan)


def _try_load_hailo() -> bool:
    if not HEF_PATH or not os.path.isfile(HEF_PATH):
        logger.info("No HEF model at %r — simulation mode", HEF_PATH)
        return False
    try:
        from hailo_platform import (  # noqa: F401
            HEF, VDevice, ConfigureParams, HailoStreamInterface,
            InputVStreamParams, OutputVStreamParams, FormatType,
        )
        hef = HEF(HEF_PATH)
        vdevice = VDevice()
        params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = vdevice.configure(hef, params)[0]
        input_name = hef.get_input_vstream_infos()[0].name
        input_shape = hef.get_input_vstream_infos()[0].shape  # (H, W, C)
        inp = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        outp = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        _state.hailo_runner = {
            "vdevice": vdevice,
            "network_group": network_group,
            "input_name": input_name,
            "input_shape": input_shape,
            "input_params": inp,
            "output_params": outp,
        }
        _state.simulation = False
        logger.info("Hailo-8 model loaded: %s (input=%s)", HEF_PATH, input_shape)
        return True
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("Hailo init failed (%s) — simulation mode", exc)
        return False
    except BaseException as exc:
        # HailoRTException (e.g. OUT_OF_PHYSICAL_DEVICES) inherits BaseException
        if "hailo" in type(exc).__module__.lower():
            logger.warning("Hailo device error (%s) — simulation mode", exc)
            return False
        raise


def _mock_inference(image_bytes: bytes) -> InferenceResponse:
    t0 = time.perf_counter()
    time.sleep(random.uniform(0.02, 0.06))  # noqa: S311
    detections = []
    for label in MOCK_LABELS:
        count = random.randint(MOCK_MIN_COUNT, MOCK_MAX_COUNT)  # noqa: S311
        for _ in range(count):
            x = random.randint(50, 1600)  # noqa: S311
            y = random.randint(50, 900)  # noqa: S311
            w = random.randint(30, 120)  # noqa: S311
            h = random.randint(30, 120)  # noqa: S311
            detections.append(Detection(
                label=label,
                score=round(random.uniform(0.5, 0.99), 2),  # noqa: S311
                box=[float(x), float(y), float(w), float(h)],
            ))
    elapsed = (time.perf_counter() - t0) * 1000
    return InferenceResponse(success=True, inference_ms=round(elapsed, 2), detections=detections)


_COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _hailo_inference(image_bytes: bytes) -> InferenceResponse:
    from hailo_platform import InferVStreams

    t0 = time.perf_counter()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return InferenceResponse(success=False, inference_ms=0.0, detections=[])

    runner = _state.hailo_runner
    if runner is None:
        return InferenceResponse(success=False, inference_ms=0.0, detections=[])

    h, w, _ = runner["input_shape"]
    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (w, h))
    input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

    with runner["network_group"].activate():
        with InferVStreams(runner["network_group"], runner["input_params"], runner["output_params"]) as vs:
            raw = vs.infer({runner["input_name"]: input_data})

    # Output: {name: list[batch]} -> batch[0] = (80, N, 5) where 5 = [y_min, x_min, y_max, x_max, score]
    detections = []
    output_key = list(raw.keys())[0]
    batch = raw[output_key][0]  # first batch item: list of 80 class arrays
    for class_id, class_dets in enumerate(batch):
        class_arr = np.array(class_dets)
        if class_arr.size == 0:
            continue
        for det in class_arr:
            score = float(det[4])
            if score < CONF_THRESHOLD:
                continue
            # Convert normalized coords to pixel coords
            y_min, x_min, y_max, x_max = det[0], det[1], det[2], det[3]
            px = float(x_min * orig_w)
            py = float(y_min * orig_h)
            pw = float((x_max - x_min) * orig_w)
            ph = float((y_max - y_min) * orig_h)
            label = _COCO_LABELS[class_id] if class_id < len(_COCO_LABELS) else str(class_id)
            detections.append(Detection(
                label=label,
                score=round(score, 3),
                box=[px, py, pw, ph],
            ))

    elapsed = (time.perf_counter() - t0) * 1000
    return InferenceResponse(success=True, inference_ms=round(elapsed, 2), detections=detections[:200])



@app.post("/inference", response_model=InferenceResponse)
async def inference(image: UploadFile = File(...)) -> InferenceResponse:  # noqa: B008
    image_bytes = await image.read()
    if _state.simulation:
        return _mock_inference(image_bytes)
    return _hailo_inference(image_bytes)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", simulation=_state.simulation, model=HEF_PATH or "mock")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104 — bind all interfaces for Docker
