"""Promotion gate: first model becomes champion; a worse model is rejected; a better one wins."""

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from taxi_demand.training.registry import register_and_gate

RCFG = {"model_name": "test-model", "min_relative_improvement": 0.01}


def _model(rounds, X, y):
    b = lgb.train({"objective": "poisson", "verbose": -1}, lgb.Dataset(X, y), rounds)
    with mlflow.start_run():
        return mlflow.lightgbm.log_model(b, name="model").model_uri, b


def test_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(0, 10, 2000)})
    y = rng.poisson(X["x"] * 3)

    from taxi_demand.training.metrics import wape

    uri1, b1 = _model(3, X, y)  # weak model
    d1 = register_and_gate(uri1, wape(y, b1.predict(X)), X, y, RCFG)
    assert d1["decision"] == "promoted"

    uri2, b2 = _model(1, X, y)  # even weaker -> rejected
    d2 = register_and_gate(uri2, wape(y, b2.predict(X)), X, y, RCFG)
    assert d2["decision"] == "rejected"

    uri3, b3 = _model(100, X, y)  # strong -> promoted
    d3 = register_and_gate(uri3, wape(y, b3.predict(X)), X, y, RCFG)
    assert d3["decision"] == "promoted"

    champ = MlflowClient().get_model_version_by_alias("test-model", "champion")
    assert champ.version == d3["new_version"]
