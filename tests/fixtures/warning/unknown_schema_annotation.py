"""DataFrame[X] annotation referencing an unresolvable schema emits pplw-unknown-schema."""

import pandera.polars as pa
from pandera.typing.polars import DataFrame
from some_external_package import ExternalSchema  # not followed


class LocalSchema(pa.DataFrameModel):
    id: int
    name: str


def passthrough(df: DataFrame[ExternalSchema]) -> DataFrame[ExternalSchema]:
    # The schema lives in a third-party package, so the function body
    # can't be checked precisely — polypolarism warns instead of silently
    # skipping the file.
    return df
