"""Valid fixture: ``pl.UInt128`` (polars 1.34+) as a Pandera schema dtype.

Exercises landmark version 1.34 — the ``UInt128`` introduction. Symmetric
with ``Int128`` (1.18) but unsigned; mostly used for hash / id columns.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class HashIndexSchema(pa.DataFrameModel):
    key: pl.UInt128
    value: pl.Int64


def passthrough(df: DataFrame[HashIndexSchema]) -> DataFrame[HashIndexSchema]:
    return df.filter(pl.col("value") > 0)


def cast_to_uint128(df: DataFrame[HashIndexSchema]) -> DataFrame[HashIndexSchema]:
    return df.with_columns(pl.col("key").cast(pl.UInt128))
