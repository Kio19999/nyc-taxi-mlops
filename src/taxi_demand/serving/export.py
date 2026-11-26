"""Export the current champion into a self-contained serving bundle.

models/champion/
  model.txt       LightGBM model
  metadata.json   version, metrics, feature list
  history.parquet last N hours of zone-hour demand (the 'online feature store' for now)

In Phase 5 this bundle goes to S3 and the container pulls it at startup.
"""

from __future__ import annotations

import json
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from taxi_demand.config import load_config, resolve
from taxi_demand.logging_utils import get_logger

log = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{resolve('mlflow.db')}"))
    name = cfg["registry"]["model_name"]
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, "champion")
    booster = mlflow.lightgbm.load_model(f"models:/{name}@champion")

    out = resolve(cfg["serving"]["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"))

    zh = pd.read_parquet(resolve(cfg["data"]["interim_dir"]) / "zone_hour.parquet")
    cutoff = zh["hour"].max() - pd.Timedelta(hours=cfg["serving"]["history_hours"] - 1)
    zh[zh["hour"] >= cutoff].to_parquet(out / "history.parquet", index=False)

    run = client.get_run(mv.run_id) if mv.run_id else None
    meta = {
        "model_name": name,
        "version": mv.version,
        "run_id": mv.run_id,
        "metrics": run.data.metrics if run else {},
        "features": cfg["training"]["features"],
        "history_end": str(zh["hour"].max()),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("exported champion v%s to %s (history to %s)", mv.version, out, meta["history_end"])


if __name__ == "__main__":
    main()
