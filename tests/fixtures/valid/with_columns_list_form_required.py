"""with_columns([...]) list-arg form adds required columns correctly (issue #97).

A FP regression: when with_columns([expr]) dropped list elements,
adding the required 'a' column via list form caused a false-positive
"Missing column 'a'" error.
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
