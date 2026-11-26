"""Integration test: one synthetic month through clean -> validate -> features."""

import numpy as np
import pandas as pd
import yaml

from taxi_demand.features.build import main


def test_pipeline_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(1)
    n = 5000
    pickup = pd.Timestamp("2024-02-01") + pd.to_timedelta(
        rng.integers(0, 29 * 24 * 60, n), unit="m"
    )
    pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=20),
            "trip_distance": 3.0,
            "fare_amount": 15.0,
            "PULocationID": rng.integers(1, 264, n).astype("int32"),
        }
    ).to_parquet(raw / "yellow_tripdata_2024-02.parquet", index=False)

    cfg = {
        "data": {
            "dataset": "yellow_tripdata",
            "start_month": "2024-02",
            "end_month": "2024-02",
            "raw_dir": str(raw),
            "interim_dir": str(tmp_path / "interim"),
            "processed_dir": str(tmp_path / "processed"),
        },
        "validation": {
            "max_drop_rate": 0.1,
            "max_trip_hours": 6,
            "max_trip_miles": 200,
            "max_fare": 1000,
        },
        "features": {"lags": [1, 24, 168], "rolling_windows": [24], "target": "pickups"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    main(cfg_path)

    zh = pd.read_parquet(tmp_path / "interim" / "zone_hour.parquet")
    feats = pd.read_parquet(tmp_path / "processed" / "features.parquet")
    assert len(zh) == 263 * 29 * 24
    assert zh["pickups"].sum() == n
    assert len(feats) == 263 * (29 * 24 - 168)
