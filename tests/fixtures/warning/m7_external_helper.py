"""M7 warning: calling an externally-imported helper emits PLW003."""

import pandera.polars as pa
from othermodule import process  # noqa: F401 — external import we can't analyse
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int


def f(df: DataFrame[S]):
    return process(df)
