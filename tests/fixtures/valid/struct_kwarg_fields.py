"""Valid: keyword args to ``pl.struct(...)`` name the struct fields (issue #47).

``pl.struct(x=pl.col("a"))`` builds ``Struct{x: Int64}``, so ``unnest`` and
``.struct.field(...)`` resolve the keyword-named field.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class AB(pa.DataFrameModel):
    a: int
    b: int

    class Config:
        coerce = True


class XOut(pa.DataFrameModel):
    x: int

    class Config:
        strict = True
        coerce = True


def kwarg_struct_unnest(df: DataFrame[AB]) -> DataFrame[XOut]:
    return df.select(w=pl.struct(x=pl.col("a"))).unnest("w")


def struct_field_access(df: DataFrame[AB]) -> DataFrame[XOut]:
    return df.select(x=pl.struct(x=pl.col("a")).struct.field("x"))


class WhenStructOut(pa.DataFrameModel):
    x: int = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


def when_struct_unnest(df: DataFrame[AB]) -> DataFrame[WhenStructOut]:
    # The shape from the original report: a struct built in a when branch,
    # then unnested. The missing otherwise pads with null, so the unnested
    # field is nullable.
    return df.select(w=pl.when(pl.col("a") > 0).then(pl.struct(x=pl.col("a")))).unnest("w")
