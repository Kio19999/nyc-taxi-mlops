"""Pipeline step: raw monthly files -> validated zone-hour table -> feature table."""

from __future__ import annotations

import json

import pandas as pd

from taxi_demand.config import DEFAULT_CONFIG, load_config, resolve
from taxi_demand.data.clean import _month_bounds, clean_and_aggregate, densify
from taxi_demand.data.ingest import month_range
from taxi_demand.data.schemas import ZoneHourSchema
from taxi_demand.features.engineering import build_features
from taxi_demand.logging_utils import get_logger

log = get_logger(__name__)
ZONES = list(range(1, 264))


def main(config_path=DEFAULT_CONFIG) -> None:
    cfg = load_config(config_path)
    dcfg, vcfg, fcfg = cfg["data"], cfg["validation"], cfg["features"]
    raw_dir, interim, processed = (
        resolve(dcfg[k]) for k in ("raw_dir", "interim_dir", "processed_dir")
    )
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    frames, reports = [], []
    for month in month_range(dcfg["start_month"], dcfg["end_month"]):
        path = raw_dir / f"{dcfg['dataset']}_{month}.parquet"
        if not path.exists():
            log.warning("missing %s, skipping (run `make data`)", path.name)
            continue
        agg, report = clean_and_aggregate(path, month, vcfg)
        start, end = _month_bounds(month)
        frames.append(densify(agg, ZONES, start, end))
        reports.append(
            {
                "month": month,
                "raw": report.raw_rows,
                "kept": report.kept_rows,
                "drop_rate": round(report.drop_rate, 4),
            }
        )

    if not frames:
        raise SystemExit("no raw files found; run `make data` first")

    zone_hour = pd.concat(frames, ignore_index=True)
    zone_hour["hour"] = zone_hour["hour"].astype("datetime64[ns]")
    ZoneHourSchema.validate(zone_hour)
    zone_hour.to_parquet(interim / "zone_hour.parquet", index=False)
    (interim / "quality_report.json").write_text(json.dumps(reports, indent=2))

    features = build_features(zone_hour, fcfg)
    features.to_parquet(processed / "features.parquet", index=False)
    log.info(
        "zone_hour rows=%d | feature rows=%d cols=%d",
        len(zone_hour),
        len(features),
        features.shape[1],
    )


if __name__ == "__main__":
    main()
