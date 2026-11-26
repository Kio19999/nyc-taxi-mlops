import numpy as np
import pandas as pd

from taxi_demand.features.engineering import add_lag_features, build_features


def _grid(hours=200):
    h = pd.date_range("2024-01-01", periods=hours, freq="h")
    return pd.DataFrame(
        {
            "zone_id": np.repeat([1, 2], hours).astype("int32"),
            "hour": np.tile(h, 2),
            "pickups": np.concatenate([np.arange(hours), np.arange(hours) * 10]),
        }
    )


def test_lags_do_not_cross_zones_or_leak():
    df = add_lag_features(_grid(), lags=[1, 24], windows=[3], target="pickups")
    z2 = df[df.zone_id == 2].reset_index(drop=True)
    assert np.isnan(z2.loc[0, "lag_1"])  # no value borrowed from zone 1
    assert z2.loc[5, "lag_1"] == 40  # previous hour: 4 * 10
    assert z2.loc[5, "roll_mean_3"] == 30  # mean of hours 2-4, excludes current


def test_build_features_drops_warmup():
    cfg = {"lags": [1, 168], "rolling_windows": [24], "target": "pickups"}
    out = build_features(_grid(), cfg)
    assert out[[c for c in out if c.startswith(("lag_", "roll_"))]].notna().all().all()
    assert {"hour_of_day", "day_of_week", "is_holiday"} <= set(out.columns)
    assert len(out) == 2 * (200 - 168)
