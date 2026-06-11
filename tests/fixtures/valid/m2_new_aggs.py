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
    # std(ddof=1) is null for singleton groups at runtime, so the column is
    # honestly nullable. polypolarism infers std() as plain Float64 (a
    # documented leniency), which still satisfies Nullable[Float64].
    sales_std: pl.Float64 = pa.Field(nullable=True)
    sales_med: pl.Float64
    units_prod: int


def stats(df: DataFrame[In]) -> DataFrame[Out]:
    return df.group_by("region").agg(
        pl.col("sales").std().alias("sales_std"),
        pl.col("sales").median().alias("sales_med"),
        pl.col("units").product().alias("units_prod"),
    )
