"""Error: Nullable type passed where non-nullable expected."""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class NonNullableSchema(pa.DataFrameModel):
    id: int
    value: int


class NullableValueSchema(pa.DataFrameModel):
    id: int
    value: int = pa.Field(nullable=True)


def expects_non_nullable(df: DataFrame[NonNullableSchema]) -> DataFrame[NonNullableSchema]:
    """Expects non-nullable Int64."""
    return df


def caller(data: DataFrame[NullableValueSchema]) -> DataFrame[NonNullableSchema]:
    """Error: value is Nullable[Int64], mismatches expected Int64."""
    return expects_non_nullable(data)
