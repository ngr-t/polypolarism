"""Valid test case: Config.coerce relaxes UInt32 vs Int64 (issue #9).

``pl.len()`` and ``n_unique()`` produce UInt32; the output schema declares
``int`` (Int64) with ``coerce = True``, so pandera casts at validation time.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class SalesSchema(pa.DataFrameModel):
    country: str
    amount: int

    class Config:
        coerce = True


class CountsSchema(pa.DataFrameModel):
    country: str
    n: int  # pl.len() yields UInt32, coerced to Int64 at runtime
    unique_amounts: int  # n_unique() yields UInt32, coerced to Int64

    class Config:
        coerce = True


def counts_by_country(sales: DataFrame[SalesSchema]) -> DataFrame[CountsSchema]:
    """Group by country; row count and distinct amounts per group."""
    return sales.group_by("country").agg(
        n=pl.len(),
        unique_amounts=pl.col("amount").n_unique(),
    )
