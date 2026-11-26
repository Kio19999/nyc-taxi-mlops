"""Model registry + champion/challenger promotion gate.

Every trained model is registered. It becomes the 'champion' (the one that gets served)
only if it beats the current champion on the SAME test data by a minimum margin.
Otherwise it is tagged 'challenger' and the champion stays.
"""

from __future__ import annotations

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from taxi_demand.logging_utils import get_logger
from taxi_demand.training.metrics import wape

log = get_logger(__name__)


def champion_wape(name: str, X_test: pd.DataFrame, y_test) -> tuple[float | None, str | None]:
    """Re-score the champion on the current test set (old runs used different data)."""
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        return None, None
    model = mlflow.lightgbm.load_model(f"models:/{name}@champion")
    return wape(y_test, model.predict(X_test)), mv.version


def register_and_gate(model_uri: str, new_wape: float, X_test, y_test, rcfg: dict) -> dict:
    name = rcfg["model_name"]
    client = MlflowClient()
    old_wape, old_version = champion_wape(name, X_test, y_test)
    mv = mlflow.register_model(model_uri, name)

    threshold = None if old_wape is None else old_wape * (1 - rcfg["min_relative_improvement"])
    promoted = old_wape is None or new_wape < threshold
    client.set_registered_model_alias(name, "champion" if promoted else "challenger", mv.version)

    decision = {
        "model_name": name,
        "new_version": mv.version,
        "new_wape": round(new_wape, 5),
        "champion_version_before": old_version,
        "champion_wape_before": None if old_wape is None else round(old_wape, 5),
        "decision": "promoted" if promoted else "rejected",
    }
    for k, v in decision.items():
        client.set_model_version_tag(name, mv.version, f"gate.{k}", str(v))
    log.info("promotion gate: %s", decision)
    return decision
