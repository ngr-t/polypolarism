"""Rolling totality through int-constant window args (backlog B-5).

``min_samples`` passed as a module-level or function-local int constant
resolves exactly like a literal: probed (polars 1.41.2), ``min_samples<=1``
fills every window (the window always contains the row itself), so the
rolling outputs below are total on non-nullable inputs and may be declared
non-nullable. ``rolling_sum`` keeps its dtype-carrying rule (Int64 stays
Int64). The false-negative twin is ``invalid/rolling_const_min_samples``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame

MIN_SAMPLES = 1


class S(pa.DataFrameModel):
    sales: pl.Float64
    qty: int


class Rolled(pa.DataFrameModel):
    sales: pl.Float64
    qty: int
    roll_mean: pl.Float64  # total: min_samples=MIN_SAMPLES (=1) fills every window
    qty_sum: int  # total + dtype-carrying: rolling_sum(Int64) -> Int64

    class Config:
        strict = True


def rolled(df: DataFrame[S]) -> DataFrame[Rolled]:
    ms = 1
    return df.with_columns(
        roll_mean=pl.col("sales").rolling_mean(window_size=3, min_samples=MIN_SAMPLES),
        qty_sum=pl.col("qty").rolling_sum(window_size=3, min_samples=ms),
    )
