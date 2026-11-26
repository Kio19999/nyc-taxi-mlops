"""Download NYC TLC trip records (monthly Parquet) into the raw zone.

Idempotent: months already downloaded are skipped, so the scheduler can run it
every month and only new files are fetched.
"""

from __future__ import annotations

from pathlib import Path

import requests

from taxi_demand.config import load_config, resolve
from taxi_demand.logging_utils import get_logger

log = get_logger(__name__)


def month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings between start and end."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def download(url: str, dest: Path, timeout: int = 120) -> bool:
    """Stream url to dest atomically. Returns False if the file isn't published (404)."""
    if dest.exists():
        log.info("skip (exists): %s", dest.name)
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        if r.status_code == 404:
            log.warning("not published yet: %s", url)
            return False
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return True


def main() -> None:
    cfg = load_config()["data"]
    raw_dir = resolve(cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    download(cfg["zone_lookup_url"], raw_dir / "taxi_zone_lookup.csv")
    for month in month_range(cfg["start_month"], cfg["end_month"]):
        fname = f"{cfg['dataset']}_{month}.parquet"
        download(f"{cfg['base_url']}/{fname}", raw_dir / fname)


if __name__ == "__main__":
    main()
