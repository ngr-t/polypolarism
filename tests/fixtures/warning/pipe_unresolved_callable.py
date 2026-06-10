"""pipe() with a callable from another package emits PLW002 (treated as identity)."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame
from some_external_package import enrich  # not analysable


class S(pa.DataFrameModel):
    id: int
    value: pl.Float64


def apply_external_pipe(df: DataFrame[S]) -> DataFrame[S]:
    # polypolarism can't see into `enrich`, so it warns and assumes identity.
    return df.pipe(enrich)
