"""a .str method called on a column that doesn't exist."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    name: str


def f(df: DataFrame[S]):
    return df.select(pl.col("missing").str.lower().alias("x"))
