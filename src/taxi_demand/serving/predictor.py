"""Loads the serving bundle once and turns (zones, hour) into predictions."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxi_demand.features.online import build_online_features


class Predictor:
    def __init__(self, model_dir: Path, fcfg: dict, features: list[str], categorical: list[str]):
        self.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        self.history = pd.read_parquet(model_dir / "history.parquet")
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self.fcfg, self.features, self.categorical = fcfg, features, categorical

    @property
    def next_hour(self) -> pd.Timestamp:
        return self.history["hour"].max() + pd.Timedelta(hours=1)

    def predict(self, zone_ids: list[int], hour) -> pd.DataFrame:
        feats = build_online_features(self.history, zone_ids, hour, self.fcfg)
        X = feats[self.features].copy()
        for c in self.categorical:
            X[c] = X[c].astype("category")
        feats["predicted_pickups"] = np.clip(self.booster.predict(X), 0, None)
        return feats[["zone_id", "hour", "predicted_pickups"]]
