"""Renderer for area_monitor mode — count overlay on camera feed."""

from __future__ import annotations

import cv2
import numpy as np

from src.display.schemas import HUDUpdate

C_WHITE = (255, 255, 255)
C_GREEN = (0, 200, 0)
C_RED = (0, 0, 220)
C_GRAY = (120, 120, 120)


def render(canvas: np.ndarray, hud: HUDUpdate) -> np.ndarray:
    h, w = canvas.shape[:2]

    # Draw detection bounding boxes
    if hud.detections:
        for det in hud.detections:
            if len(det.box) == 4:
                x, y_pos, bw, bh = [int(v) for v in det.box]
                cv2.rectangle(canvas, (x, y_pos), (x + bw, y_pos + bh), C_GREEN, 2)
                label = f"{det.label} {det.score:.2f}"
                cv2.putText(canvas, label, (x, y_pos - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GREEN, 2)

    total = hud.total_count if hud.total_count is not None else 0
    delta = hud.delta if hud.delta is not None else 0
    alert = hud.alert or False
    location = hud.location or ""
    last_updated = hud.last_updated or ""

    color = C_RED if alert else C_GREEN
    state_text = "ALERT" if alert else "MONITORING"

    # Semi-transparent top bar
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    cv2.putText(canvas, state_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    if location:
        cv2.putText(canvas, location, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GRAY, 2)

    # Count display (top right)
    count_text = str(total)
    ts = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)[0]
    cv2.putText(canvas, count_text, (w - ts[0] - 30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

    # Delta
    if delta != 0:
        sign = "+" if delta > 0 else ""
        delta_text = f"{sign}{delta}"
        delta_color = C_GREEN if delta > 0 else C_RED
        cv2.putText(canvas, delta_text, (w - 150, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, delta_color, 2)

    # Timestamp bottom
    if last_updated:
        overlay2 = canvas.copy()
        cv2.rectangle(overlay2, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, canvas, 0.4, 0, canvas)
        cv2.putText(canvas, f"Updated: {last_updated}", (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_GRAY, 2)

    # Border
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 4)

    return canvas
