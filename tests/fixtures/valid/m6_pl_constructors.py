"""M6: pl.struct / pl.concat_str / pl.coalesce / pl.format expression constructors."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    first: str
    last: str
    primary: pl.Float64 = pa.Field(nullable=True)
    fallback: pl.Float64


def make_record(df: DataFrame[S]):
    return df.select(
        pl.concat_str([pl.col("first"), pl.col("last")], separator=" ").alias("full_name"),
        pl.coalesce(pl.col("primary"), pl.col("fallback")).alias("amount"),
        pl.struct(pl.col("first"), pl.col("last")).alias("name"),
    )
