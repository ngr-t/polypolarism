"""Invalid: a kwarg-built struct field with a wrong declared dtype must fail.

False-negative twin of ``valid/struct_kwarg_fields.py`` (issue #47): when
``pl.struct(x=...)`` silently produced ``Struct{}``, the valid fixture still
passed via leniency — only this wrong-declaration twin can prove the field
dtype is actually checked. ``pl.struct(x=pl.col("a"))`` builds
``Struct{x: Int64}``, so declaring the unnested ``x`` as ``str`` is an error.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class AB(pa.DataFrameModel):
    a: int
    b: int


class XStrOut(pa.DataFrameModel):
    x: str

    class Config:
        strict = True


def kwarg_struct_unnest_wrong_dtype(df: DataFrame[AB]) -> DataFrame[XStrOut]:
    return df.select(w=pl.struct(x=pl.col("a"))).unnest("w")


def struct_field_access_wrong_dtype(df: DataFrame[AB]) -> DataFrame[XStrOut]:
    return df.select(x=pl.struct(x=pl.col("a")).struct.field("x"))
