"""Feature engineering for hourly demand forecasting.

Rule: every feature for hour t uses only data from hours < t (no leakage).
"""

from __future__ import annotations

import holidays
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    h = out["hour"]
    out["hour_of_day"] = h.dt.hour.astype("int8")
    out["day_of_week"] = h.dt.day_of_week.astype("int8")
    out["month"] = h.dt.month.astype("int8")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    years = range(h.dt.year.min(), h.dt.year.max() + 1)
    us_holidays = holidays.US(years=years, subdiv="NY")
    out["is_holiday"] = h.dt.date.map(lambda d: d in us_holidays).astype("int8")
    return out


def add_lag_features(
    df: pd.DataFrame, lags: list[int], windows: list[int], target: str
) -> pd.DataFrame:
    """Per-zone lags and rolling means. Assumes a dense, sorted zone x hour grid."""
    out = df.sort_values(["zone_id", "hour"]).copy()
    g = out.groupby("zone_id", sort=False)[target]
    for lag in lags:
        out[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)  # rolling stats end at t-1 so the current hour never leaks in
    for w in windows:
        out[f"roll_mean_{w}"] = (
            shifted.groupby(out["zone_id"], sort=False)
            .rolling(w, min_periods=w)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return out


def build_features(zone_hour: pd.DataFrame, fcfg: dict) -> pd.DataFrame:
    df = add_calendar_features(zone_hour)
    df = add_lag_features(df, fcfg["lags"], fcfg["rolling_windows"], fcfg["target"])
    warmup = max(max(fcfg["lags"]), max(fcfg["rolling_windows"]) + 1)
    feature_cols = [c for c in df.columns if c.startswith(("lag_", "roll_mean_"))]
    df = df.dropna(subset=feature_cols)
    assert len(df) > 0, f"not enough history: need > {warmup} hours per zone"
    return df.reset_index(drop=True)
