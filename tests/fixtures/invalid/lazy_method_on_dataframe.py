"""Invalid test case: lazy-only method called on an eager DataFrame (pple-lazy-only-method)."""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class InputSchema(pa.DataFrameModel):
    id: int
    name: str


class OutputSchema(pa.DataFrameModel):
    id: int
    name: str


def collect_eager_frame(df: DataFrame[InputSchema]) -> DataFrame[OutputSchema]:
    """ERROR: .collect() is only available on LazyFrame."""
    return df.collect()
