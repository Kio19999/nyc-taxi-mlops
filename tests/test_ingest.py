from taxi_demand.data.ingest import month_range


def test_month_range_crosses_year():
    assert month_range("2023-11", "2024-02") == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_month_range_single():
    assert month_range("2024-05", "2024-05") == ["2024-05"]
