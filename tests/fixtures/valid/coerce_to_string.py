"""Valid: ``Config.coerce`` tolerates always-castable-to-String columns (issue #58).

Direction-aware coercibility: pandera's coerce casts the inferred dtype
into the declared one, and casting any probed-formattable dtype into
String is value-independent — ``pl.len()`` (UInt32) and ``max()`` (Int64)
columns declared ``str`` under ``coerce = True`` pass at runtime. The
boundary stays pinned by ``invalid/coerce_limits.py``: the reverse
Utf8 -> Int64 direction is value-dependent and must keep failing.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class SalesSchema(pa.DataFrameModel):
    country: str
    amount: int

    class Config:
        coerce = True


class CountsAsText(pa.DataFrameModel):
    country: str
    n: str  # pl.len() yields UInt32 -> String: always-castable under coerce
    top_amount: str  # max() keeps Int64 -> String: always-castable too

    class Config:
        coerce = True


def counts_as_text(sales: DataFrame[SalesSchema]) -> DataFrame[CountsAsText]:
    return sales.group_by("country").agg(
        n=pl.len(),
        top_amount=pl.col("amount").max(),
    )
