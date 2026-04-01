"""Renderer for batch_count mode — detection boxes + count overlay."""

from __future__ import annotations

import cv2
import numpy as np

from src.display.schemas import HUDUpdate

C_WHITE = (255, 255, 255)
C_GREEN = (0, 200, 0)
C_RED = (0, 0, 220)
C_YELLOW = (0, 220, 220)
C_GRAY = (120, 120, 120)


def render(canvas: np.ndarray, hud: HUDUpdate) -> np.ndarray:
    h, w = canvas.shape[:2]

    # Draw detection bounding boxes
    if hud.detections:
        for det in hud.detections:
            if len(det.box) == 4:
                x, y, bw, bh = [int(v) for v in det.box]
                cv2.rectangle(canvas, (x, y), (x + bw, y + bh), C_GREEN, 2)
                label = f"{det.label} {det.score:.2f}"
                cv2.putText(canvas, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GREEN, 2)

    state = hud.state or "IDLE"
    sku = hud.sku or ""
    live = hud.live_count if hud.live_count is not None else 0
    target = hud.target_count if hud.target_count is not None else 0
    result = hud.result

    # State banner color
    if state == "IDLE":
        color = C_YELLOW
        text = "IDLE — Scan QR"
    elif state == "COUNTING":
        color = C_YELLOW
        text = f"COUNTING: {sku}"
    elif result == "PASS":
        color = C_GREEN
        text = "PASS"
    elif result == "FAIL":
        color = C_RED
        text = "FAIL"
    else:
        color = C_GRAY
        text = state

    # Semi-transparent top bar
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    # State text
    cv2.putText(canvas, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # SKU
    if sku:
        cv2.putText(canvas, f"SKU: {sku}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GRAY, 2)

    # Count display (top right)
    count_text = f"{live}/{target}"
    ts = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    cv2.putText(canvas, count_text, (w - ts[0] - 30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, C_WHITE, 3)

    # Border
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 4)

    return canvas
