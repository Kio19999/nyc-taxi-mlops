"""Online (request-time) feature computation.

Must produce exactly the same values as the offline pipeline in engineering.py.
tests/test_online_features.py enforces this (training/serving skew check).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxi_demand.features.engineering import add_calendar_features


class FeatureError(ValueError):
    """Requested hour can't be served from the available history."""


def build_online_features(
    history: pd.DataFrame, zone_ids: list[int], hour, fcfg: dict
) -> pd.DataFrame:
    hour = pd.Timestamp(hour).floor("h")
    first, last = history["hour"].min(), history["hour"].max()
    lookback = pd.Timedelta(hours=max(max(fcfg["lags"]), max(fcfg["rolling_windows"])))
    if hour > last + pd.Timedelta(hours=1):
        raise FeatureError(f"can only forecast up to {last + pd.Timedelta(hours=1)}")
    if hour - lookback < first:
        raise FeatureError(
            f"not enough history before {hour}; earliest servable is {first + lookback}"
        )

    series = history[history["zone_id"].isin(zone_ids)].set_index(["zone_id", "hour"])["pickups"]
    one_h = pd.Timedelta(hours=1)
    rows = []
    for z in zone_ids:
        s = series.loc[z] if z in series.index.get_level_values(0) else pd.Series(dtype=float)
        row = {"zone_id": z, "hour": hour}
        for lag in fcfg["lags"]:
            row[f"lag_{lag}"] = float(s.get(hour - pd.Timedelta(hours=lag), np.nan))
        for w in fcfg["rolling_windows"]:
            window = s.loc[hour - pd.Timedelta(hours=w) : hour - one_h]
            row[f"roll_mean_{w}"] = float(window.mean()) if len(window) == w else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df["zone_id"] = df["zone_id"].astype("int32")
    return add_calendar_features(df)
