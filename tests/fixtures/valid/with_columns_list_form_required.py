"""with_columns non-literal argument forms add columns correctly (issue #97).

A FP regression: when ``with_columns`` dropped the columns produced by a
list / starred-spread / dict-unpacking argument, adding a *required* column
through one of those forms caused a false-positive "Missing column 'a'".
All forms produce ``{k, v, a}`` at runtime, identical to the varargs form,
so each must register the new column.
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


class KVa(pa.DataFrameModel):
    k: str
    v: float
    a: float

    class Config:
        strict = True
        coerce = False


@pa.check_types
def fp_listform_required(df: DataFrame[KV]) -> DataFrame[KVa]:
    return df.with_columns([(pl.col("v") * 2).alias("a")])


@pa.check_types
def fp_starform_required(df: DataFrame[KV]) -> DataFrame[KVa]:
    return df.with_columns(*[(pl.col("v") * 2).alias("a")])


@pa.check_types
def fp_kwargs_required(df: DataFrame[KV]) -> DataFrame[KVa]:
    return df.with_columns(**{"a": pl.col("v") * 2})
