"""Error: Untyped function body cannot be analyzed."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class IdSchema(pa.DataFrameModel):
    id: int


def external_function(df):
    """External library call, cannot be analyzed."""
    return some_external_lib.process(df)


def caller(data: DataFrame[IdSchema]) -> DataFrame[IdSchema]:
    """Error when untyped function body cannot be analyzed."""
    return external_function(data)
