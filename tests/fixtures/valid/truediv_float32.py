"""Valid: Float32 / Float32 true division keeps Float32 (issue #135).

True division widens to Float64 for every operand pair EXCEPT ``Float32 /
Float32``, which polars keeps at Float32. Declaring the Float32 result must
type-check; a mixed ``Float32 / Int64`` still widens to Float64.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    f32: pl.Float32
    i64: pl.Int64


class Out(pa.DataFrameModel):
    same: pl.Float32  # f32 / f32 -> Float32
    mixed: pl.Float64  # f32 / i64 -> Float64


def truediv(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        same=pl.col("f32") / pl.col("f32"),
        mixed=pl.col("f32") / pl.col("i64"),
    )
