"""Data contracts. Downstream steps can trust anything that passes these."""

import pandera.pandas as pa

ZoneHourSchema = pa.DataFrameSchema(
    {
        "zone_id": pa.Column("int32", pa.Check.in_range(1, 263)),
        "hour": pa.Column("datetime64[ns]"),
        "pickups": pa.Column("int32", pa.Check.ge(0)),
    },
    unique=["zone_id", "hour"],
    strict=True,
    coerce=False,
)
