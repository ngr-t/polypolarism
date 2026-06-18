"""filter predicate has a non-boolean dtype (issue #28, pple-non-boolean-predicate)."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class AInt(pa.DataFrameModel):
    a: int

    class Config:
        coerce = True


@pa.check_types
def bug_filter_nonbool(df: DataFrame[AInt]) -> DataFrame[AInt]:
    return df.filter(pl.col("a"))  # 'a' is Int64, not Boolean
