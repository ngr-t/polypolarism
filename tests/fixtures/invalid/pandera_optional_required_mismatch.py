"""Declared return requires a column that is only optional in the input."""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class InSchema(pa.DataFrameModel):
    id: int
    age: int | None


class OutSchema(pa.DataFrameModel):
    id: int
    age: int  # required — but input may be missing it


def passthrough(df: DataFrame[InSchema]) -> DataFrame[OutSchema]:
    return df
