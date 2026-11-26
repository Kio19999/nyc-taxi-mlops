"""Flow logic: export runs only when the gate promotes a new champion."""

import importlib.util
from pathlib import Path

import pytest
from prefect.testing.utilities import prefect_test_harness

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def harness():
    with prefect_test_harness():
        yield


def _load_flow():
    spec = importlib.util.spec_from_file_location(
        "training_flow", ROOT / "pipelines/training_flow.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("decision,expect_export", [("promoted", True), ("rejected", False)])
def test_export_only_on_promotion(monkeypatch, decision, expect_export):
    mod = _load_flow()
    calls = []
    monkeypatch.setattr(mod.ingest, "main", lambda: calls.append("ingest"))
    monkeypatch.setattr(mod.build, "main", lambda: calls.append("features"))
    monkeypatch.setattr(mod.export, "main", lambda: calls.append("export"))
    monkeypatch.setattr(
        mod.train,
        "main",
        lambda: {
            "lightgbm": {"test_wape": 0.15},
            "gate": {"decision": decision, "new_version": 2, "champion_version_before": 1},
        },
    )
    out = mod.training_pipeline()
    assert out["decision"] == decision
    assert ("export" in calls) is expect_export
    assert calls[:2] == ["ingest", "features"]
