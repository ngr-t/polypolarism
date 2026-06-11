"""upsample — identity columns, but non-key columns become Nullable (issue #74).

Probed (polars 1.41.2, identical on 1.37.0): the gap rows that upsample
inserts are null-filled in every column except the keys — the time column
keeps its dtype and stays non-null, ``group_by`` columns are filled per
group and stay non-null, everything else gains nulls (the docstring's own
example chains ``fill_null(strategy="forward")`` for exactly this reason).
False-negative twin: ``invalid/upsample_nonnullable_declared``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Readings(pa.DataFrameModel):
    t: pl.Datetime
    g: str
    v: int


class Upsampled(pa.DataFrameModel):
    t: pl.Datetime
    g: str = pa.Field(nullable=True)
    v: int = pa.Field(nullable=True)

    class Config:
        strict = True


class UpsampledPerGroup(pa.DataFrameModel):
    t: pl.Datetime
    g: str
    v: int = pa.Field(nullable=True)

    class Config:
        strict = True


def half_hourly(data: DataFrame[Readings]) -> DataFrame[Upsampled]:
    """Every non-key column is nullable in the upsampled frame."""
    return data.sort("t").upsample(time_column="t", every="30m")


def half_hourly_per_group(data: DataFrame[Readings]) -> DataFrame[UpsampledPerGroup]:
    """group_by keys are filled per group — they stay non-nullable."""
    return data.sort("g", "t").upsample(time_column="t", every="30m", group_by="g")
