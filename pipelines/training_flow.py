"""End-to-end training pipeline as a Prefect flow.

    ingest -> features (clean + validate + build) -> train (+ promotion gate) -> export

Each step is a Prefect task: retries on failure, logged, visible in the Prefect UI.
Export only runs when the gate promotes a new champion, so a worse model never ships.

Run once:      make flow
Run monthly:   make flow-serve   (5th of each month, 06:00 — TLC publishes with a lag)
"""

from __future__ import annotations

from prefect import flow, get_run_logger, task

from taxi_demand.data import ingest
from taxi_demand.features import build
from taxi_demand.serving import export
from taxi_demand.training import train


@task(retries=3, retry_delay_seconds=60, name="ingest")
def ingest_task() -> None:
    ingest.main()


@task(name="features")
def features_task() -> None:
    build.main()  # raises DataQualityError / SchemaError -> flow fails loudly, nothing trains


@task(name="train-and-gate")
def train_task() -> dict:
    return train.main()


@task(name="export-champion")
def export_task() -> None:
    export.main()


@flow(name="taxi-demand-training", log_prints=True)
def training_pipeline() -> dict:
    log = get_run_logger()
    ingest_task()
    features_task()
    results = train_task()
    gate = results["gate"]
    lgbm = results["lightgbm"]
    log.info(
        "model v%s test WAPE=%.4f -> %s", gate["new_version"], lgbm["test_wape"], gate["decision"]
    )
    if gate["decision"] == "promoted":
        export_task()
        log.info("new champion exported; rebuild/redeploy the API image to ship it")
    else:
        log.info("champion unchanged (v%s); nothing to deploy", gate["champion_version_before"])
    return {
        "decision": gate["decision"],
        "version": gate["new_version"],
        "test_wape": lgbm["test_wape"],
    }


if __name__ == "__main__":
    import sys

    if "--serve" in sys.argv:
        training_pipeline.serve(name="monthly-retrain", cron="0 6 5 * *")
    else:
        training_pipeline()
