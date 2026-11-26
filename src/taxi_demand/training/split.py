from __future__ import annotations

import pandas as pd


def time_split(df: pd.DataFrame, train_end: str, valid_end: str):
    """Chronological split. Returns (train, valid, test)."""
    t1, t2 = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    train = df[df["hour"] < t1]
    valid = df[(df["hour"] >= t1) & (df["hour"] < t2)]
    test = df[df["hour"] >= t2]
    return train, valid, test
