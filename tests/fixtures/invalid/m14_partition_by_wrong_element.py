"""partition_by elements keep the input schema — wrong element declarations.

False-negative twin of ``valid/m14_partition_by.py``: same subscript and
for-loop element flows, but the declared element schemas disagree with the
input dtypes.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    region: str
    sales: pl.Float64


class ElemWrong(pa.DataFrameModel):
    region: str
    sales: pl.Int64  # WRONG: partition elements keep the input dtype Float64


def first_partition(df: DataFrame[S]) -> DataFrame[ElemWrong]:
    parts = df.partition_by("region")
    return parts[0]


class SalesWrong(pa.DataFrameModel):
    sales: pl.Int64  # WRONG: the loop element's 'sales' stays Float64


def per_partition(df: DataFrame[S]) -> DataFrame[SalesWrong]:
    accum = df.select(pl.col("sales"))
    for part in df.partition_by("region", include_key=False):
        accum = part.select(pl.col("sales"))
    return accum
