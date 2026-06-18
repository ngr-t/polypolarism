"""map_elements without return_dtype= emits pplw-missing-return-dtype."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int
    value: pl.Float64


def f(df: DataFrame[S]):
    # No return_dtype — polypolarism falls back to the receiver dtype and warns.
    return df.with_columns(pl.col("value").map_elements(lambda v: v * 2).alias("v2"))
