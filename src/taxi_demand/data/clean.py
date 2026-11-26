"""Clean raw trips and aggregate to zone x hour demand with DuckDB.

Row-level rules run in SQL (scales to hundreds of millions of rows on a laptop);
the small aggregated output is then checked against a Pandera contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from taxi_demand.logging_utils import get_logger

log = get_logger(__name__)


class DataQualityError(RuntimeError):
    """Raised when a batch fails quality gates; the pipeline must stop."""


@dataclass
class QualityReport:
    month: str
    raw_rows: int
    kept_rows: int

    @property
    def drop_rate(self) -> float:
        return 1 - self.kept_rows / self.raw_rows if self.raw_rows else 1.0


def _month_bounds(month: str) -> tuple[str, str]:
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    return str(start), str(end)


def clean_and_aggregate(path: Path, month: str, vcfg: dict) -> tuple[pd.DataFrame, QualityReport]:
    """Return hourly pickup counts per zone for one monthly file, plus a quality report."""
    start, end = _month_bounds(month)
    con = duckdb.connect()
    src = f"read_parquet('{path.as_posix()}')"
    raw_rows = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]

    where = f"""
        tpep_pickup_datetime >= TIMESTAMP '{start}'
        AND tpep_pickup_datetime <  TIMESTAMP '{end}'
        AND tpep_dropoff_datetime > tpep_pickup_datetime
        AND date_diff('minute', tpep_pickup_datetime, tpep_dropoff_datetime)
            <= {vcfg["max_trip_hours"]} * 60
        AND trip_distance > 0 AND trip_distance <= {vcfg["max_trip_miles"]}
        AND fare_amount >= 0 AND fare_amount <= {vcfg["max_fare"]}
        AND PULocationID BETWEEN 1 AND 263
    """
    kept_rows = con.execute(f"SELECT count(*) FROM {src} WHERE {where}").fetchone()[0]
    agg = con.execute(
        f"""
        SELECT CAST(PULocationID AS INTEGER)              AS zone_id,
               date_trunc('hour', tpep_pickup_datetime)   AS hour,
               CAST(count(*) AS INTEGER)                  AS pickups
        FROM {src}
        WHERE {where}
        GROUP BY 1, 2
        """
    ).df()
    con.close()

    report = QualityReport(month, raw_rows, kept_rows)
    log.info("%s: raw=%d kept=%d drop=%.2f%%", month, raw_rows, kept_rows, 100 * report.drop_rate)
    if report.drop_rate > vcfg["max_drop_rate"]:
        raise DataQualityError(
            f"{month}: drop rate {report.drop_rate:.1%} exceeds {vcfg['max_drop_rate']:.0%}"
        )
    return agg, report


def densify(agg: pd.DataFrame, zones: list[int], start: str, end: str) -> pd.DataFrame:
    """Add explicit zero rows for zone-hours with no pickups (models need a regular grid)."""
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    grid = pd.MultiIndex.from_product([zones, hours], names=["zone_id", "hour"]).to_frame(
        index=False
    )
    out = grid.merge(agg, on=["zone_id", "hour"], how="left")
    out["pickups"] = out["pickups"].fillna(0).astype("int32")
    out["zone_id"] = out["zone_id"].astype("int32")
    return out.sort_values(["zone_id", "hour"]).reset_index(drop=True)
