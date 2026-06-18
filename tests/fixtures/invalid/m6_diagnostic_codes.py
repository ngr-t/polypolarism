"""ensure errors carry [pple-*] codes."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int


def f(df: DataFrame[S]):
    return df.select(pl.col("missing"))
