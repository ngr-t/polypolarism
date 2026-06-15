"""with_columns([...]) list-arg form must be treated as varargs (issue #97).

A strict schema rejects extra columns; using the list form to add a column
must register the column — not silently drop it.  The FP case (strict
schema expects 'a' but list-form adds it → false-positive "Missing column")
is covered by the complementary valid fixture.
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
