"""M7 warning: df.pipe(lambda ...) emits PLW004 — lambdas can't be analysed."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int
    value: pl.Float64


def f(df: DataFrame[S]) -> DataFrame[S]:
    return df.pipe(lambda d: d.filter(pl.col("value") > 0))
