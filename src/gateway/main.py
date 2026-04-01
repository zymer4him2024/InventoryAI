"""Gateway Agent — FastAPI app and route handlers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.gateway import config
from src.gateway.loops import inference_loop, qr_scan_loop
from src.gateway.schemas import JobRequest, JobResponse, StatusResponse, HealthResponse
from src.gateway.state import gw

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

config.validate()


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    asyncio.create_task(inference_loop())
    asyncio.create_task(qr_scan_loop())
    logger.info("Gateway started — APP_ID=%s DEVICE_ID=%s", config.APP_ID, config.DEVICE_ID)
    yield
    if gw.http_client and not gw.http_client.is_closed:
        await gw.http_client.aclose()


app = FastAPI(title=f"InventoryAI Gateway ({config.APP_ID})", lifespan=_lifespan)


@app.post("/job", response_model=JobResponse)
async def create_job(req: JobRequest) -> JobResponse:
    async with gw.mode_lock:
        await gw.mode.handle_qr(req.sku)
        state = gw.mode.get_state()
    return JobResponse(status="ok", sku=req.sku, state=state)


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    async with gw.mode_lock:
        display_state = await gw.mode.get_display_state()
        state = gw.mode.get_state()
    return StatusResponse(
        app_id=config.APP_ID,
        device_id=config.DEVICE_ID,
        state=state,
        display=display_state,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", app_id=config.APP_ID, device_id=config.DEVICE_ID)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104 — bind all interfaces for Docker
