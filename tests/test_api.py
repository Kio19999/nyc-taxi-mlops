"""API tests against a tiny real model bundle built in a temp dir."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from taxi_demand.config import load_config
from taxi_demand.features.engineering import build_features


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = load_config()
    feats, cats = cfg["training"]["features"], cfg["training"]["categorical"]
    rng = np.random.default_rng(0)
    hours = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    zones = list(range(1, 264))
    hist = pd.DataFrame(
        {
            "zone_id": np.repeat(zones, len(hours)).astype("int32"),
            "hour": np.tile(hours, len(zones)),
            "pickups": rng.poisson(5, len(zones) * len(hours)).astype("int32"),
        }
    )
    df = build_features(hist, cfg["features"])
    X = df[feats].copy()
    X["zone_id"] = X["zone_id"].astype("category")
    booster = lgb.train(
        {"objective": "poisson", "verbose": -1},
        lgb.Dataset(X, df["pickups"], categorical_feature=cats),
        num_boost_round=5,
    )
    booster.save_model(str(tmp_path / "model.txt"))
    hist.to_parquet(tmp_path / "history.parquet", index=False)
    (tmp_path / "metadata.json").write_text(json.dumps({"version": "test"}))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    from taxi_demand.serving.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_default_next_hour_all_zones(client):
    r = client.post("/predict", json={})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 263
    assert all(p["predicted_pickups"] >= 0 for p in body["predictions"])
    assert "X-Process-Time-ms" in r.headers


def test_predict_specific_zones(client):
    r = client.post("/predict", json={"zone_ids": [132, 161]})
    assert [p["zone_id"] for p in r.json()["predictions"]] == [132, 161]


def test_invalid_zone_and_future_hour_rejected(client):
    assert client.post("/predict", json={"zone_ids": [999]}).status_code == 422
    assert client.post("/predict", json={"hour": "2030-01-01T00:00:00"}).status_code == 422
