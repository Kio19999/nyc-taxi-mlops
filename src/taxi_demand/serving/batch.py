"""Batch scoring: forecast every zone for the next hour and write a Parquet file.

This is how a dispatch dashboard would consume forecasts: one scheduled job,
no API calls. Same Predictor as the API, so batch and online can't diverge.
"""

from __future__ import annotations

import os
from pathlib import Path

from taxi_demand.config import load_config, resolve
from taxi_demand.logging_utils import get_logger
from taxi_demand.serving.predictor import Predictor

log = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    model_dir = Path(os.getenv("MODEL_DIR", resolve(cfg["serving"]["model_dir"])))
    p = Predictor(
        model_dir, cfg["features"], cfg["training"]["features"], cfg["training"]["categorical"]
    )
    out = p.predict(list(range(1, 264)), p.next_hour)
    out["model_version"] = str(p.metadata["version"])
    dest = resolve("data/predictions")
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"forecast_{p.next_hour:%Y%m%d_%H}.parquet"
    out.to_parquet(path, index=False)
    log.info("wrote %d forecasts for %s -> %s", len(out), p.next_hour, path.name)
    top = out.nlargest(5, "predicted_pickups")[["zone_id", "predicted_pickups"]]
    log.info("top zones:\n%s", top.to_string(index=False))


if __name__ == "__main__":
    main()
