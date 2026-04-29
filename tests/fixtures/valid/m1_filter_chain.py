"""M1: filter / sort / head are identity-typed; chains preserve schema."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int
    value: pl.Float64


def top_positive(df: DataFrame[S]) -> DataFrame[S]:
    return df.filter(pl.col("value") > 0).sort("value").head(10)
