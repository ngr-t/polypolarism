"""with_columns non-literal argument forms must be treated as varargs (issue #97).

A strict schema rejects extra columns; adding a column through a list /
starred-spread / dict-unpacking argument must register the column — not
silently drop it (which was a false negative: the strict violation passed).
The complementary FP cases (a required column added through these forms) are
in the valid fixture.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True
        coerce = False


@pa.check_types
def fn_listform_extra(df: DataFrame[KV]) -> DataFrame[KV]:
    return df.with_columns([(pl.col("v") * 2).alias("a")])


@pa.check_types
def fn_starform_extra(df: DataFrame[KV]) -> DataFrame[KV]:
    return df.with_columns(*[(pl.col("v") * 2).alias("a")])


@pa.check_types
def fn_kwargs_extra(df: DataFrame[KV]) -> DataFrame[KV]:
    return df.with_columns(**{"a": pl.col("v") * 2})
