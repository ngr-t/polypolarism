"""Valid: bare string column names in select / with_columns (issue #7)."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    b: str
    c: pl.Float64


class Out(pa.DataFrameModel):
    a: int
    b: str


def select_positional_strings(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select("a", "b")


def select_string_list(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(["a", "b"])


def select_mixed_string_and_expr(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select("a", pl.col("b"))


def with_columns_string_reselect(df: DataFrame[In]) -> DataFrame[In]:
    return df.with_columns("a")
