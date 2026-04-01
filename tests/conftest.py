"""Shared fixtures for InventoryAI tests."""

import os

# Set required env vars before any src imports
os.environ.setdefault("APP_ID", "batch_count")
os.environ.setdefault("DEVICE_ID", "TEST-001")
os.environ.setdefault("FIREBASE_SIMULATE", "true")
os.environ.setdefault("SIMULATE_CAMERA", "true")
