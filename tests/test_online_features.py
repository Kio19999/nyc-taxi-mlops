"""Training/serving skew test: online features must equal offline features exactly."""

import numpy as np
import pandas as pd
import pytest

from taxi_demand.features.engineering import build_features
from taxi_demand.features.online import FeatureError, build_online_features

FCFG = {"lags": [1, 2, 24, 168], "rolling_windows": [24, 168], "target": "pickups"}


def _history(hours=400):
    rng = np.random.default_rng(3)
    h = pd.date_range("2024-03-01", periods=hours, freq="h")
    return pd.DataFrame(
        {
            "zone_id": np.repeat([1, 2, 3], hours).astype("int32"),
            "hour": np.tile(h, 3),
            "pickups": rng.poisson(20, 3 * hours).astype("int32"),
        }
    )


def test_online_matches_offline():
    hist = _history()
    offline = build_features(hist, FCFG)
    target_hour = hist["hour"].iloc[300]
    online = build_online_features(hist, [1, 2, 3], target_hour, FCFG)
    off = offline[offline["hour"] == target_hour].sort_values("zone_id").reset_index(drop=True)
    cols = [c for c in online.columns if c not in ("zone_id", "hour")]
    pd.testing.assert_frame_equal(online[cols], off[cols], check_dtype=False)


def test_next_hour_is_servable_but_further_is_not():
    hist = _history()
    nxt = hist["hour"].max() + pd.Timedelta(hours=1)
    assert build_online_features(hist, [1], nxt, FCFG)["lag_1"].notna().all()
    with pytest.raises(FeatureError):
        build_online_features(hist, [1], nxt + pd.Timedelta(hours=1), FCFG)


def test_insufficient_history_rejected():
    with pytest.raises(FeatureError):
        build_online_features(_history(), [1], "2024-03-02 00:00", FCFG)
