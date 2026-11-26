import numpy as np
import pandas as pd

from taxi_demand.training.metrics import mae, wape
from taxi_demand.training.split import time_split


def test_wape_and_mae():
    y, p = np.array([10, 0, 10]), np.array([8, 1, 12])
    assert mae(y, p) == 5 / 3
    assert wape(y, p) == 5 / 20


def test_time_split_is_chronological_and_disjoint():
    df = pd.DataFrame({"hour": pd.date_range("2024-01-01", periods=24 * 120, freq="h")})
    tr, va, te = time_split(df, "2024-03-01", "2024-04-01")
    assert tr["hour"].max() < va["hour"].min() <= va["hour"].max() < te["hour"].min()
    assert len(tr) + len(va) + len(te) == len(df)
