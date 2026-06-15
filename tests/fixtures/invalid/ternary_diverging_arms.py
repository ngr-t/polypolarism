"""Ternary return with diverging arm schemas must fail (issue #94).

The if-arm adds column 'a' but the else-arm adds column 'b' instead.
Both arms are checked against the declared return type KVa (which
requires 'a').
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True
        coerce = True


class KVa(pa.DataFrameModel):
    k: str
    v: float
    a: float

    class Config:
        strict = True
        coerce = True


@pa.check_types
def ternary_diverge(df: DataFrame[KV], flag: bool) -> DataFrame[KVa]:
    return df.with_columns(a=pl.col("v") * 2) if flag else df.with_columns(b=pl.col("v"))
