"""Error: Missing required column in function argument."""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class TwoColSchema(pa.DataFrameModel):
    id: int
    name: str


class IdOnlySchema(pa.DataFrameModel):
    id: int


def requires_two_columns(df: DataFrame[TwoColSchema]) -> DataFrame[TwoColSchema]:
    """Function that requires two columns."""
    return df


def caller(data: DataFrame[IdOnlySchema]) -> DataFrame[TwoColSchema]:
    """Error: missing column 'name'."""
    return requires_two_columns(data)
