"""Gateway configuration — loads and validates all env vars at startup."""

from __future__ import annotations

import os
import sys

_VALID_APP_IDS = {"batch_count", "bundle_check", "area_monitor"}

APP_ID = os.getenv("APP_ID", "")
DEVICE_ID = os.getenv("DEVICE_ID", "")
CAMERA_URL = os.getenv("CAMERA_URL", "http://camera_agent:8002")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference_agent:8001")
DISPLAY_URL = os.getenv("DISPLAY_URL", "http://display_agent:8003")
FIREBASE_SYNC_URL = os.getenv("FIREBASE_SYNC_URL", "http://firebase_sync_agent:8004")

# batch_count
COUNTING_WINDOW_SEC = float(os.getenv("COUNTING_WINDOW_SEC", "5"))
COUNT_TOLERANCE = int(os.getenv("COUNT_TOLERANCE", "0"))

# bundle_check
BUNDLE_TIMEOUT_SEC = float(os.getenv("BUNDLE_TIMEOUT_SEC", "10"))

# area_monitor
LOW_STOCK_THRESHOLD = int(os.getenv("LOW_STOCK_THRESHOLD", "5"))
HIGH_STOCK_THRESHOLD = int(os.getenv("HIGH_STOCK_THRESHOLD", "50"))
AREA_SNAPSHOT_INTERVAL_SEC = float(os.getenv("AREA_SNAPSHOT_INTERVAL_SEC", "30"))
LOCATION_NAME = os.getenv("LOCATION_NAME", "Unknown")

# Timing
INFERENCE_INTERVAL_SEC = float(os.getenv("INFERENCE_INTERVAL_SEC", "0.5"))
QR_SCAN_INTERVAL_SEC = float(os.getenv("QR_SCAN_INTERVAL_SEC", "1.0"))
RESULT_HOLD_SEC = float(os.getenv("RESULT_HOLD_SEC", "3.0"))
HEALTH_TIMEOUT = 5.0


def validate() -> None:
    if APP_ID not in _VALID_APP_IDS:
        print(f"FATAL: APP_ID={APP_ID!r} invalid. Must be one of {sorted(_VALID_APP_IDS)}", file=sys.stderr)
        sys.exit(1)
    if not DEVICE_ID:
        print("FATAL: DEVICE_ID is not set", file=sys.stderr)
        sys.exit(1)
