"""False-negative twin of ``valid/temporal_reductions`` (issue #85).

Pins both failure families probed on polars 1.41.2:

- mean preserves the receiver INSTANCE, so declaring the wrong time unit
  for a grouped Datetime mean must fail statically (pandera's runtime
  validation rejects the unit mismatch too).
- The genuinely-invalid temporal cells stay pple-groupby: ``var`` on Duration
  raises InvalidOperationError in both contexts; ``sum``/``std`` on
  Datetime raise as whole-frame reductions, while their grouped forms
  silently yield an unconditionally all-null column — the non-nullable
  declared output makes the runtime side fail validation there.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Sessions(pa.DataFrameModel):
    device: str
    started_at: pl.Datetime("ms", time_zone="UTC")
    duration: pl.Duration("ns")


class WrongUnit(pa.DataFrameModel):
    device: str
    # WRONG: mean preserves the receiver's ms unit (probed 1.41.2)
    mean_started: pl.Datetime("us", time_zone="UTC")

    class Config:
        strict = True


def mean_wrong_unit(df: DataFrame[Sessions]) -> DataFrame[WrongUnit]:
    return df.group_by("device").agg(pl.col("started_at").mean().alias("mean_started"))


class VarOut(pa.DataFrameModel):
    device: str
    v: pl.Duration("ns") = pa.Field(nullable=True)


def var_on_duration(df: DataFrame[Sessions]) -> DataFrame[VarOut]:
    # ERROR: var is unsupported for Duration — InvalidOperationError in
    # BOTH contexts (probed 1.41.2), unlike std which Duration supports.
    return df.group_by("device").agg(pl.col("duration").var().alias("v"))


class SumOut(pa.DataFrameModel):
    device: str
    total: pl.Datetime("ms", time_zone="UTC")


def sum_on_datetime_grouped(df: DataFrame[Sessions]) -> DataFrame[SumOut]:
    # ERROR: grouped sum on Datetime never raises at runtime — it silently
    # yields an all-null column of the receiver dtype (probed 1.41.2); the
    # non-nullable declared output catches it at validation time.
    return df.group_by("device").agg(pl.col("started_at").sum().alias("total"))


class StdOut(pa.DataFrameModel):
    s: pl.Datetime("ms", time_zone="UTC") = pa.Field(nullable=True)


def std_on_datetime_select(df: DataFrame[Sessions]) -> DataFrame[StdOut]:
    # ERROR: the whole-frame (select) std on Datetime raises
    # InvalidOperationError at runtime (probed 1.41.2).
    return df.select(pl.col("started_at").std().alias("s"))
