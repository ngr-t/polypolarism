"""Valid: comparisons, is_in and casts that polars accepts (issues #33/#34).

Contrasts for the PLY009/PLY013 false-negative fixes: int == int,
int < float (and an int literal against a Float64 column), str.is_in
on a list of strings, List(Int64) -> List(Float64) cast, the
value-dependent Utf8 -> Int64 cast, and an Int64 -> Date cast (numeric
to temporal is allowed — probed against polars 1.41.2).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    b: int
    x: pl.Float64
    s: str
    epoch_day: int
    vals: pl.List(pl.Int64) = pa.Field()


class Out(pa.DataFrameModel):
    same: bool
    below: bool
    pos: bool
    tagged: bool
    parsed: int
    day: pl.Date
    ratios: pl.List(pl.Float64) = pa.Field()


def compare_and_cast(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        same=pl.col("a") == pl.col("b"),
        below=pl.col("a") < pl.col("x"),
        pos=pl.col("x") > 0,
        tagged=pl.col("s").is_in(["x", "y"]),
        parsed=pl.col("s").cast(pl.Int64),
        day=pl.col("epoch_day").cast(pl.Date),
        ratios=pl.col("vals").cast(pl.List(pl.Float64)),
    )
