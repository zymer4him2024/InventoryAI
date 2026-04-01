"""Firebase Sync Agent — async Firestore writes and SKU config lookups."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from src.firebase_sync.schemas import SKUConfig, WriteRequest, WriteResponse, HealthResponse

logger = logging.getLogger("firebase_sync")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "surgicalai01")
FIREBASE_SIMULATE = os.getenv("FIREBASE_SIMULATE", "false").lower() == "true"
EVENTS_LOG_PATH = Path(os.getenv("EVENTS_LOG_PATH", "/app/data/events.jsonl"))

try:
    from google.cloud.exceptions import GoogleCloudError
except ImportError:
    GoogleCloudError = OSError  # type: ignore[misc,assignment]

app = FastAPI(title="InventoryAI Firebase Sync Agent")

_db = None
_simulation = True


def _init_firebase() -> bool:
    global _db, _simulation
    if FIREBASE_SIMULATE:
        logger.info("Firebase simulation mode (FIREBASE_SIMULATE=true)")
        return False

    cred_path = FIREBASE_CREDENTIALS_PATH
    if not cred_path or not os.path.isfile(cred_path):
        logger.warning("Firebase credentials not found at %r — simulation mode", cred_path)
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {"projectId": FIREBASE_PROJECT_ID})
        _db = firestore.client()
        _simulation = False
        logger.info("Firebase initialized (project=%s)", FIREBASE_PROJECT_ID)
        return True
    except (ImportError, ValueError, FileNotFoundError) as exc:
        logger.warning("Firebase init failed (%s) — simulation mode", exc)
        return False


@app.on_event("startup")
async def _startup() -> None:
    _init_firebase()
    EVENTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Firebase Sync Agent started (simulation=%s)", _simulation)


@app.post("/write", response_model=WriteResponse)
async def write_event(req: WriteRequest) -> WriteResponse | JSONResponse:
    data = {**req.data, "written_at": datetime.now(timezone.utc).isoformat()}

    if _simulation:
        with open(EVENTS_LOG_PATH, "a") as f:
            f.write(json.dumps({"collection": req.collection, **data}) + "\n")
        logger.info("[SIM] Wrote to %s: %s", req.collection, json.dumps(data)[:200])
        return WriteResponse(status="simulated", collection=req.collection)

    try:
        _db.collection(req.collection).add(data)
        logger.info("Wrote to Firestore %s", req.collection)
        return WriteResponse(status="ok", collection=req.collection)
    except GoogleCloudError as exc:
        logger.error("Firestore write failed: %s", exc)
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)


@app.post("/load_sku")
async def load_sku(sku: str = Query(..., min_length=1)) -> JSONResponse:
    if _simulation:
        mock = SKUConfig(
            sku=sku,
            part_class="bolt_m6",
            target_count=10,
            required_classes=["bolt_m6", "washer_m6", "nut_m6"],
            customer_id="sim_customer",
            tolerance=0,
        )
        logger.info("[SIM] Loaded SKU config: %s", sku)
        return JSONResponse(mock.model_dump())

    try:
        doc = _db.collection("inventory_skus").document(sku).get()
        if not doc.exists:
            return JSONResponse({"error": f"SKU {sku} not found"}, status_code=404)
        sku_config = SKUConfig(**doc.to_dict())
        return JSONResponse(sku_config.model_dump())
    except GoogleCloudError as exc:
        logger.error("Firestore read failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", simulation=_simulation)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)  # noqa: S104 — bind all interfaces for Docker
