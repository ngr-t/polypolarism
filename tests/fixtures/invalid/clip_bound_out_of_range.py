"""Invalid: clip bound outside the column dtype's range always raises (issue #156).

Unlike fill_null / shift, ``clip`` never widens the result dtype — so a literal
bound that cannot be represented in the (integer) column dtype is a provable
``InvalidOperationError`` on every call ("conversion from i32 to i8 failed").
Either bound (lower or upper) is enough; a negative bound against an unsigned
column is out of range too. In-range bounds stay valid (see the valid twin).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    i8: pl.Int8
    u8: pl.UInt8


class Out(pa.DataFrameModel):
    r: pl.Int8


def clip_lower_oob(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(r=pl.col("i8").clip(-1000, 5))  # -1000 out of Int8 range


def clip_upper_oob(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(r=pl.col("i8").clip(0, 1000))  # 1000 out of Int8 range


def clip_unsigned_negative(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(r=pl.col("u8").clip(-1, 5))  # -1 out of UInt8 range
