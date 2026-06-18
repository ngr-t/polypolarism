"""Invalid: duplicate output column names within one projection (issue #36).

``df.select(pl.col("a"), pl.col("b").alias("a"))`` produces two outputs
named ``a`` — polars raises ``DuplicateError`` at runtime. polypolarism
flags it statically as pple-duplicate-column.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    b: pl.Float64


class Out(pa.DataFrameModel):
    a: pl.Float64


def project(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(pl.col("a"), pl.col("b").alias("a"))
