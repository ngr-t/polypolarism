"""M2: filter predicates with is_not_null, comparison and logical ops are typed."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int
    value: pl.Float64 = pa.Field(nullable=True)


def keep_valid(df: DataFrame[S]) -> DataFrame[S]:
    return df.filter(pl.col("value").is_not_null() & (pl.col("id") > 0))
