"""passing LazyFrame[S] where DataFrame[S] is expected — pple-eager-lazy-mismatch."""

import pandera.polars as pa
from pandera.typing.polars import DataFrame, LazyFrame


class S(pa.DataFrameModel):
    id: int


def helper(df: DataFrame[S]) -> DataFrame[S]:
    return df


def caller(lf: LazyFrame[S]):
    return helper(lf)
