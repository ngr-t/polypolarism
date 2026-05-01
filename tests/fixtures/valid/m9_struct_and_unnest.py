"""struct.field access + df.unnest flatten the Struct column."""

from typing import Annotated

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    id: int
    user: Annotated[pl.Struct, {"name": pl.Utf8(), "age": pl.Int64()}]


class Picked(pa.DataFrameModel):
    id: int
    name: str
    age: int


def access_field(df: DataFrame[In]) -> DataFrame[Picked]:
    return df.select(
        pl.col("id"),
        pl.col("user").struct.field("name").alias("name"),
        pl.col("user").struct.field("age").alias("age"),
    )


class Unnested(pa.DataFrameModel):
    id: int
    name: str
    age: int


def unnest_struct(df: DataFrame[In]) -> DataFrame[Unnested]:
    return df.unnest("user")
