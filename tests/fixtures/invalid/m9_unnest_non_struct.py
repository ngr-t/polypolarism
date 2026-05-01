"""unnest on a non-Struct column."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    v: pl.Float64


def f(df: DataFrame[S]):
    return df.unnest("v")
