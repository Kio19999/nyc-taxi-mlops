import pandas as pd
import pandera.pandas as pa
import pytest

from taxi_demand.data.clean import DataQualityError, clean_and_aggregate, densify
from taxi_demand.data.schemas import ZoneHourSchema


def test_bad_rows_filtered(raw_trips, vcfg):
    path, n_good = raw_trips
    agg, report = clean_and_aggregate(path, "2024-01", vcfg)
    assert report.raw_rows == n_good + 2
    assert report.kept_rows == n_good
    assert agg["pickups"].sum() == n_good


def test_quality_gate_fails(raw_trips, vcfg):
    path, _ = raw_trips
    with pytest.raises(DataQualityError):
        clean_and_aggregate(path, "2024-01", {**vcfg, "max_drop_rate": 0.001})


def test_densify_and_contract(raw_trips, vcfg):
    path, n_good = raw_trips
    agg, _ = clean_and_aggregate(path, "2024-01", vcfg)
    dense = densify(agg, [132, 161, 237], "2024-01-01", "2024-01-03")
    assert len(dense) == 3 * 48
    assert dense["pickups"].sum() == n_good
    dense["hour"] = dense["hour"].astype("datetime64[ns]")
    ZoneHourSchema.validate(dense)


def test_contract_rejects_negative():
    df = pd.DataFrame(
        {
            "zone_id": pd.array([1], dtype="int32"),
            "hour": pd.to_datetime(["2024-01-01"]),
            "pickups": pd.array([-1], dtype="int32"),
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        ZoneHourSchema.validate(df)
