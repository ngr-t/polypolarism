"""Valid: casting right after an unmodeled method retracts pplw-unmodeled-method.

Remedy twin of ``warning/unmodeled_method``: the explicit ``.cast(...)``
is exactly the repair the warning recommends, so the chain checks
precisely (Int8 column, no warning, no leniency note).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    v: int


class OutPinned(pa.DataFrameModel):
    v: int
    flag: pl.Int8


def peaks_pinned(df: DataFrame[In]) -> DataFrame[OutPinned]:
    return df.with_columns(flag=pl.col("v").peak_max().cast(pl.Int8))
