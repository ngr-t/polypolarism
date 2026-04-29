"""Untyped function passthrough: body analysis infers return type."""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class IdSchema(pa.DataFrameModel):
    id: int


def untyped_passthrough(df):
    """No type annotation, return DataFrame as-is."""
    return df


def caller(data: DataFrame[IdSchema]) -> DataFrame[IdSchema]:
    """Type is inferred via body analysis even through untyped function."""
    return untyped_passthrough(data)
