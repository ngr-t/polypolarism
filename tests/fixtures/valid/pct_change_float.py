"""``pct_change`` divides — int input yields Float64, floats keep width (issue #71).

Probed (polars 1.41.2): ``pct_change`` on any int width returns Float64
(it divides by the previous element); a Float32 receiver keeps Float32.
The head slot is always null, so the result is nullable. Before #71 the
method was classified shift-like (dtype-preserving) and these correct
declarations were falsely rejected.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Ints(pa.DataFrameModel):
    a: int


class Floats(pa.DataFrameModel):
    f: pl.Float32


class PctF64(pa.DataFrameModel):
    x: pl.Float64 = pa.Field(nullable=True)


class PctF32(pa.DataFrameModel):
    x: pl.Float32 = pa.Field(nullable=True)


def int_pct_change_is_float64(df: DataFrame[Ints]) -> DataFrame[PctF64]:
    return df.select(x=pl.col("a").pct_change())


def float32_pct_change_keeps_width(df: DataFrame[Floats]) -> DataFrame[PctF32]:
    return df.select(x=pl.col("f").pct_change())
