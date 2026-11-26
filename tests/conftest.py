import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_trips(tmp_path):
    """A tiny synthetic TLC file with known good and bad rows."""
    rng = np.random.default_rng(0)
    n = 200
    pickup = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 48 * 60, n), unit="m")
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.Timedelta(minutes=15),
            "trip_distance": 2.5,
            "fare_amount": 12.0,
            "PULocationID": rng.choice([132, 161, 237], n).astype("int32"),
        }
    )
    bad = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                pd.Timestamp("2023-12-31 23:00"),
                pd.Timestamp("2024-01-01 05:00"),
            ],
            "tpep_dropoff_datetime": [
                pd.Timestamp("2024-01-01 00:10"),
                pd.Timestamp("2024-01-01 04:00"),
            ],
            "trip_distance": [1.0, 1.0],
            "fare_amount": [5.0, 5.0],
            "PULocationID": np.array([132, 132], dtype="int32"),
        }
    )  # outside month, and dropoff before pickup
    path = tmp_path / "yellow_tripdata_2024-01.parquet"
    pd.concat([df, bad]).to_parquet(path, index=False)
    return path, n


@pytest.fixture
def vcfg():
    return {"max_drop_rate": 0.10, "max_trip_hours": 6, "max_trip_miles": 200, "max_fare": 1000}
