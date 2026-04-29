"""M3 invalid: a .str method called on a column that doesn't exist."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    name: str


def f(df: DataFrame[S]):
    return df.select(pl.col("missing").str.lower().alias("x"))
