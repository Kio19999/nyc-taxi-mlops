"""Real-time prediction API.

GET  /health       liveness probe for ECS / load balancer
GET  /model-info   which model version is live and how good it is
POST /predict      {"hour": "...", "zone_ids": [132, 161]}  (omit zone_ids = all zones)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from taxi_demand.config import load_config, resolve
from taxi_demand.features.online import FeatureError
from taxi_demand.logging_utils import get_logger
from taxi_demand.serving.predictor import Predictor

log = get_logger("taxi_demand.api")
ALL_ZONES = list(range(1, 264))
state: dict = {}


class PredictRequest(BaseModel):
    hour: datetime | None = Field(None, description="Target hour; default = next hour after data")
    zone_ids: list[int] | None = Field(None, min_length=1, max_length=263)


class ZonePrediction(BaseModel):
    zone_id: int
    predicted_pickups: float


class PredictResponse(BaseModel):
    model_version: str
    hour: datetime
    predictions: list[ZonePrediction]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    model_dir = Path(os.getenv("MODEL_DIR", resolve(cfg["serving"]["model_dir"])))
    state["predictor"] = Predictor(
        model_dir, cfg["features"], cfg["training"]["features"], cfg["training"]["categorical"]
    )
    log.info("loaded model v%s from %s", state["predictor"].metadata["version"], model_dir)
    yield
    state.clear()


app = FastAPI(title="NYC Taxi Demand API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    log.info("%s %s %d %.1fms", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    p: Predictor = state["predictor"]
    return {**p.metadata, "next_forecastable_hour": str(p.next_hour)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    p: Predictor = state["predictor"]
    zones = req.zone_ids or ALL_ZONES
    bad = [z for z in zones if not 1 <= z <= 263]
    if bad:
        raise HTTPException(422, f"invalid zone_ids: {bad}")
    hour = req.hour or p.next_hour
    try:
        out = p.predict(zones, hour)
    except FeatureError as e:
        raise HTTPException(422, str(e)) from e
    return PredictResponse(
        model_version=str(p.metadata["version"]),
        hour=out["hour"].iloc[0],
        predictions=[
            ZonePrediction(zone_id=int(z), predicted_pickups=round(float(v), 2))
            for z, v in zip(out["zone_id"], out["predicted_pickups"], strict=True)
        ],
    )
