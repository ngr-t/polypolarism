"""DataFrame[mod.X] qualified annotation that doesn't resolve emits pplw-unknown-schema."""

import pandera.polars as pa
import some_external_package  # not followed
from pandera.typing.polars import DataFrame


class LocalSchema(pa.DataFrameModel):
    id: int
    name: str


def passthrough(
    df: DataFrame[some_external_package.RemoteSchema],
) -> DataFrame[some_external_package.RemoteSchema]:
    # The qualified schema lives in a third-party package, so the body
    # can't be checked precisely — polypolarism warns instead of silently
    # skipping the function (issue #68).
    return df
