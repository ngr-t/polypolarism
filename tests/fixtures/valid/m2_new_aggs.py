"""std / var / median / quantile / product in group_by aggregation."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    region: str
    sales: pl.Float64
    units: int


class Out(pa.DataFrameModel):
    region: str
    sales_std: pl.Float64
    sales_med: pl.Float64
    units_prod: int


def stats(df: DataFrame[In]) -> DataFrame[Out]:
    return df.group_by("region").agg(
        pl.col("sales").std().alias("sales_std"),
        pl.col("sales").median().alias("sales_med"),
        pl.col("units").product().alias("units_prod"),
    )
