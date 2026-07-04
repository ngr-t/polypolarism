"""Invalid: fill_null(<wider expr>) declared at the narrow receiver dtype (issue #158).

A wider column fill argument widens the result to the supertype, so declaring
the receiver's narrow dtype is a silent false negative — pandera rejects the
wider column at validation time. Counterpart to the valid twin, which declares
the true widened dtype.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    v8: pl.Int8 = pa.Field(nullable=True)
    i64: pl.Int64


class NarrowOut(pa.DataFrameModel):
    r: pl.Int8  # wrong: v8.fill_null(i64) is actually Int64


def fill_wider_declared_narrow(df: DataFrame[In]) -> DataFrame[NarrowOut]:
    return df.select(r=pl.col("v8").fill_null(pl.col("i64")))  # Int64, not Int8
