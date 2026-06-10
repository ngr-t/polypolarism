"""over() partition column doesn't exist (issue #32, PLY001)."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Mixed(pa.DataFrameModel):
    a: int
    s: str

    class Config:
        coerce = True


class OverO(pa.DataFrameModel):
    a: int
    s: str
    g: int

    class Config:
        strict = True
        coerce = True


@pa.check_types
def bug_over_nonexistent(df: DataFrame[Mixed]) -> DataFrame[OverO]:
    return df.with_columns(g=pl.col("a").sum().over("ghost"))  # 'ghost' doesn't exist
