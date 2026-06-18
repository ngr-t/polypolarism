"""namespace accessor on a column of the wrong dtype (issue #31, pple-wrong-namespace-dtype)."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Mixed(pa.DataFrameModel):
    a: int
    s: str

    class Config:
        coerce = True


class BoolO(pa.DataFrameModel):
    m: bool

    class Config:
        strict = True
        coerce = True


class IntO(pa.DataFrameModel):
    r: int

    class Config:
        strict = True
        coerce = True


@pa.check_types
def bug_str_on_int(df: DataFrame[Mixed]) -> DataFrame[BoolO]:
    return df.select(m=pl.col("a").str.contains("x"))  # 'a' is Int64, not String


@pa.check_types
def bug_dt_on_int(df: DataFrame[Mixed]) -> DataFrame[IntO]:
    return df.select(r=pl.col("a").dt.year())  # 'a' is Int64, not temporal


@pa.check_types
def bug_list_on_int(df: DataFrame[Mixed]) -> DataFrame[IntO]:
    return df.select(r=pl.col("a").list.sum())  # 'a' is Int64, not List/Array
