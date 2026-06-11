"""Invalid: rolling outputs declared non-nullable (issue #57).

False-negative twin of ``valid/m5_window_and_rolling.py``: rolling windows
leave rows null until ``min_samples`` (default: ``window_size``) values
are present (probed, polars 1.41.2 — any non-empty frame gets leading
nulls), so a rolling output column must be declared
``pa.Field(nullable=True)``. The non-nullable declarations below fail
statically (inferred ``Float64?`` / ``Int64?`` cannot satisfy a
non-nullable slot) and at runtime (pandera rejects the leading nulls).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    region: str
    sales: pl.Float64
    qty: int


class RollMeanNonNull(pa.DataFrameModel):
    region: str
    sales: pl.Float64
    qty: int
    roll: pl.Float64  # WRONG: rolling_mean(window_size=3) has leading nulls


def rolling_mean_nonnullable_decl(df: DataFrame[S]) -> DataFrame[RollMeanNonNull]:
    return df.with_columns(roll=pl.col("sales").rolling_mean(window_size=3))


class RollSumNonNull(pa.DataFrameModel):
    region: str
    sales: pl.Float64
    qty: int
    qty_sum: int  # WRONG: rolling_sum(window_size=3) has leading nulls


def rolling_sum_nonnullable_decl(df: DataFrame[S]) -> DataFrame[RollSumNonNull]:
    return df.with_columns(qty_sum=pl.col("qty").rolling_sum(window_size=3))
