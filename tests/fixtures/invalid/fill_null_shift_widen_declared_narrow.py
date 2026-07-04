"""Invalid: a widening fill_null / shift literal declared at the narrow dtype (issue #156).

``fill_null`` / ``shift(fill_value=)`` with a literal that does not fit the
column widen the result (``Int8`` filled with ``1000`` -> Int16), so declaring
the un-widened ``Int8`` is a silent false negative — pandera rejects the Int16
column at validation time. Counterpart to the valid twin, which declares the
true widened dtype.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    i8: pl.Int8
    v8: pl.Int8 = pa.Field(nullable=True)


class NarrowOut(pa.DataFrameModel):
    r: pl.Int8  # wrong: the value is actually Int16


def fill_null_widen_narrow(df: DataFrame[In]) -> DataFrame[NarrowOut]:
    return df.select(r=pl.col("v8").fill_null(1000))  # Int16, not Int8


def shift_widen_narrow(df: DataFrame[In]) -> DataFrame[NarrowOut]:
    return df.select(r=pl.col("i8").shift(1, fill_value=1000))  # Int16, not Int8
