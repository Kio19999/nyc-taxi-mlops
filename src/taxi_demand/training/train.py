"""Train baselines + LightGBM, evaluate on a held-out time window, log everything to MLflow.

MLflow target: $MLFLOW_TRACKING_URI if set (e.g. http://localhost:5001 after `make up`),
otherwise a local SQLite file (mlflow.db); artifacts go to ./mlartifacts.
"""

from __future__ import annotations

import json
import os

import lightgbm as lgb
import mlflow
import pandas as pd

from taxi_demand.config import DEFAULT_CONFIG, load_config, resolve
from taxi_demand.logging_utils import get_logger
from taxi_demand.training.metrics import all_metrics, wape
from taxi_demand.training.registry import register_and_gate
from taxi_demand.training.split import time_split

log = get_logger(__name__)


def _prep(df: pd.DataFrame, features: list[str], cats: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    return X


def per_zone_wape(test: pd.DataFrame, pred, top_n: int = 20) -> pd.DataFrame:
    t = test[["zone_id", "pickups"]].assign(pred=pred)
    rows = [
        {"zone_id": z, "pickups": g["pickups"].sum(), "wape": wape(g["pickups"], g["pred"])}
        for z, g in t.groupby("zone_id")
    ]
    return pd.DataFrame(rows).sort_values("pickups", ascending=False).head(top_n)


def main(config_path=DEFAULT_CONFIG) -> dict:
    cfg = load_config(config_path)
    tcfg, target = cfg["training"], cfg["features"]["target"]
    df = pd.read_parquet(resolve(cfg["data"]["processed_dir"]) / "features.parquet")
    train, valid, test = time_split(df, tcfg["train_end"], tcfg["valid_end"])
    log.info("train=%d valid=%d test=%d", len(train), len(valid), len(test))

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{resolve('mlflow.db')}"))
    mlflow.set_experiment(tcfg["experiment_name"])

    feats, cats = tcfg["features"], tcfg["categorical"]
    results = {}

    # 1) Baselines: any real model must beat these, or it isn't worth deploying.
    for name, col in [("naive_last_hour", "lag_1"), ("seasonal_naive_week", "lag_168")]:
        with mlflow.start_run(run_name=name):
            m = all_metrics(test[target], test[col], prefix="test_")
            mlflow.log_param("model_type", name)
            mlflow.log_metrics(m)
            results[name] = m
            log.info("%-22s test WAPE=%.4f MAE=%.3f", name, m["test_wape"], m["test_mae"])

    # 2) LightGBM global model across all zones.
    with mlflow.start_run(run_name="lightgbm") as run:
        dtrain = lgb.Dataset(_prep(train, feats, cats), train[target], categorical_feature=cats)
        dvalid = lgb.Dataset(_prep(valid, feats, cats), valid[target], reference=dtrain)
        booster = lgb.train(
            tcfg["lgbm_params"],
            dtrain,
            num_boost_round=tcfg["num_boost_round"],
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(tcfg["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        pred_valid = booster.predict(
            _prep(valid, feats, cats), num_iteration=booster.best_iteration
        )
        pred_test = booster.predict(_prep(test, feats, cats), num_iteration=booster.best_iteration)

        m = {
            **all_metrics(valid[target], pred_valid, prefix="valid_"),
            **all_metrics(test[target], pred_test, prefix="test_"),
        }
        best_name = min(results, key=lambda k: results[k]["test_wape"])
        base = results[best_name]["test_wape"]
        m["test_wape_improvement_vs_best_baseline"] = (base - m["test_wape"]) / base
        mlflow.log_param("best_baseline", best_name)

        mlflow.log_params(
            {
                **tcfg["lgbm_params"],
                "best_iteration": booster.best_iteration,
                "features": ",".join(feats),
                "train_end": tcfg["train_end"],
                "valid_end": tcfg["valid_end"],
            }
        )
        mlflow.log_metrics(m)

        imp = pd.DataFrame(
            {"feature": booster.feature_name(), "gain": booster.feature_importance("gain")}
        ).sort_values("gain", ascending=False)
        zones = per_zone_wape(test, pred_test)
        mlflow.log_text(imp.to_csv(index=False), "reports/feature_importance.csv")
        mlflow.log_text(zones.to_csv(index=False), "reports/top_zones_wape.csv")
        # Keep only the trees up to the early-stopping point, so the saved model == evaluated model.
        final = lgb.Booster(model_str=booster.model_to_string(num_iteration=booster.best_iteration))
        info = mlflow.lightgbm.log_model(final, name="model")

        model_dir = resolve(tcfg["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        final.save_model(str(model_dir / "model.txt"))
        (model_dir / "metrics.json").write_text(
            json.dumps({**m, "run_id": run.info.run_id}, indent=2)
        )

        results["lightgbm"] = m
        log.info(
            "lightgbm  best_iter=%d  test WAPE=%.4f MAE=%.3f  (%.1f%% better than %s)",
            booster.best_iteration,
            m["test_wape"],
            m["test_mae"],
            100 * m["test_wape_improvement_vs_best_baseline"],
            best_name,
        )

    # 3) Registry + promotion gate (outside the run so a gate failure never hides the run).
    results["gate"] = register_and_gate(
        info.model_uri, m["test_wape"], _prep(test, feats, cats), test[target], cfg["registry"]
    )
    return results


if __name__ == "__main__":
    main()
