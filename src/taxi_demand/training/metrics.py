"""Forecast metrics. WAPE is the headline: robust to the many near-zero zones."""

from __future__ import annotations

import numpy as np


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def wape(y, p) -> float:
    """Weighted absolute percentage error = sum|err| / sum|actual|."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - p).sum() / denom) if denom else float("nan")


def all_metrics(y, p, prefix: str = "") -> dict[str, float]:
    return {f"{prefix}mae": mae(y, p), f"{prefix}rmse": rmse(y, p), f"{prefix}wape": wape(y, p)}
