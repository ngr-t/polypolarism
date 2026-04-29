"""M4 invalid: explode on a column that isn't List[T]."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    v: pl.Float64


def f(df: DataFrame[S]):
    return df.explode("v")
