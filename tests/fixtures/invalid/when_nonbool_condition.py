"""pl.when condition has a non-boolean dtype (issue #37, PLY008).

Probed (polars 1.41.2): ``pl.when(pl.col("a"))`` with ``a: Int64`` raises
``SchemaError: invalid series dtype: expected `Boolean`, got `i64```.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class AInt(pa.DataFrameModel):
    a: int

    class Config:
        coerce = True


class Out(pa.DataFrameModel):
    a: int
    flag: int

    class Config:
        coerce = True


@pa.check_types
def bug_when_nonbool(df: DataFrame[AInt]) -> DataFrame[Out]:
    return df.with_columns(flag=pl.when(pl.col("a")).then(1).otherwise(0))
