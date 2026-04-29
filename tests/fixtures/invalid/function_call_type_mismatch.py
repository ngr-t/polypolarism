"""Error: Column type mismatch in function argument."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class IntIdSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


class StrIdSchema(pa.DataFrameModel):
    id: str
    value: pl.Float64


def expects_int_id(df: DataFrame[IntIdSchema]) -> DataFrame[IntIdSchema]:
    """Function that expects id: Int64."""
    return df


def caller(data: DataFrame[StrIdSchema]) -> DataFrame[IntIdSchema]:
    """Error: id type is Utf8, mismatches expected Int64."""
    return expects_int_id(data)
