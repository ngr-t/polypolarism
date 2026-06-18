"""Invalid: comparison / is_in between incompatible dtypes (issue #33).

``pl.col("s") == pl.col("a")`` compares a String with an Int64 — polars
raises ``ComputeError: cannot compare string with numeric type`` at
runtime. ``pl.col("a").is_in(["x", "y"])`` checks for String values in
Int64 data — polars raises ``InvalidOperationError``. polypolarism flags
both statically as pple-incompatible-operands.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Mixed(pa.DataFrameModel):
    a: int
    s: str


class BoolOut(pa.DataFrameModel):
    m: bool


def bug_str_eq_int(df: DataFrame[Mixed]) -> DataFrame[BoolOut]:
    return df.select(m=pl.col("s") == pl.col("a"))


def bug_is_in_type_mismatch(df: DataFrame[Mixed]) -> DataFrame[BoolOut]:
    return df.select(m=pl.col("a").is_in(["x", "y"]))
