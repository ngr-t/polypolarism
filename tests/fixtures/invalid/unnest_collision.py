"""Invalid: unnest field-name collisions that always raise DuplicateError (issue #145).

polars raises ``DuplicateError`` unconditionally when an unnested struct field
collides — with another struct's field, or with a pre-existing column. Both
variants were written into the schema with a silent dict overwrite, so the
always-crashing code passed with zero diagnostics (false negative).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class TwoStructIn(pa.DataFrameModel):
    s1: pl.Struct({"a": pl.Int64})
    s2: pl.Struct({"a": pl.Int64})


class ColAndStruct(pa.DataFrameModel):
    a: int
    s: pl.Struct({"a": pl.Int64})


class AOut(pa.DataFrameModel):
    a: int


def unnest_two_structs(df: DataFrame[TwoStructIn]) -> DataFrame[AOut]:
    return df.unnest("s1", "s2")


def unnest_collides_existing(df: DataFrame[ColAndStruct]) -> DataFrame[AOut]:
    return df.unnest("s")
