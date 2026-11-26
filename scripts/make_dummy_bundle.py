"""Create a tiny but valid serving bundle in models/champion/.

Used by CI to build and smoke-test the Docker image without real data or MLflow.
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxi_demand.config import load_config, resolve
from taxi_demand.features.engineering import build_features


def main() -> None:
    cfg = load_config()
    feats, cats = cfg["training"]["features"], cfg["training"]["categorical"]
    hours = pd.date_range("2024-12-20", periods=24 * 10, freq="h")
    zones = list(range(1, 264))
    rng = np.random.default_rng(0)
    hist = pd.DataFrame(
        {
            "zone_id": np.repeat(zones, len(hours)).astype("int32"),
            "hour": np.tile(hours, len(zones)),
            "pickups": rng.poisson(5, len(zones) * len(hours)).astype("int32"),
        }
    )
    df = build_features(hist, cfg["features"])
    X = df[feats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    booster = lgb.train(
        {"objective": "poisson", "verbose": -1},
        lgb.Dataset(X, df["pickups"], categorical_feature=cats),
        5,
    )
    out = resolve(cfg["serving"]["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"))
    hist.to_parquet(out / "history.parquet", index=False)
    (out / "metadata.json").write_text(json.dumps({"version": "ci-dummy"}))
    print(f"dummy bundle written to {out}")


if __name__ == "__main__":
    main()
