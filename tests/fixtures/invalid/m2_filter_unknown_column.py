"""M2 invalid: filter predicate references a column that doesn't exist."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int


def f(df: DataFrame[S]) -> DataFrame[S]:
    return df.filter(pl.col("missing").is_not_null())
