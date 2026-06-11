"""Invalid: int-constant min_samples that does NOT fill the window (B-5).

False-negative twin of ``valid/rolling_const_min_samples.py``: the same
rolling calls, but the constants resolve to 2 — probed (polars 1.41.2),
rows whose window holds fewer than ``min_samples`` values are null, so the
outputs are nullable and the non-nullable declarations below must fail. If
int-constant resolution ever degrades (treating any bound name as total or
falling back to Unknown), these wrong declarations would start passing.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame

MIN_SAMPLES = 2


class S(pa.DataFrameModel):
    sales: pl.Float64
    qty: int


class Rolled(pa.DataFrameModel):
    sales: pl.Float64
    qty: int
    roll_mean: pl.Float64  # WRONG: min_samples=MIN_SAMPLES (=2) leaves a leading null
    qty_sum: int  # WRONG: min_samples=ms (=2) leaves a leading null

    class Config:
        strict = True


def rolled(df: DataFrame[S]) -> DataFrame[Rolled]:
    ms = 2
    return df.with_columns(
        roll_mean=pl.col("sales").rolling_mean(window_size=3, min_samples=MIN_SAMPLES),
        qty_sum=pl.col("qty").rolling_sum(window_size=3, min_samples=ms),
    )
