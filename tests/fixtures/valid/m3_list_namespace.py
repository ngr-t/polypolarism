"""M3: pl.col(...).list methods dispatch on element/list dtype."""

from typing import Annotated

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    user_id: int
    scores: Annotated[pl.List, pl.Float64()]


class Out(pa.DataFrameModel):
    user_id: int
    score_sum: pl.Float64
    score_count: pl.UInt32
    top_score: pl.Float64


def summarize(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        pl.col("user_id"),
        pl.col("scores").list.sum().alias("score_sum"),
        pl.col("scores").list.len().alias("score_count"),
        pl.col("scores").list.max().alias("top_score"),
    )
