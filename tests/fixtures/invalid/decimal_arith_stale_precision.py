"""Invalid: declaring the pre-growth Decimal precision (issue #52).

``Decimal(10,2) + Decimal(10,2)`` materializes as ``Decimal(38, 2)``
(probed, polars 1.41.2) — declaring the stale input precision
``Decimal(10, 2)`` for the sum is wrong and pandera would reject the
frame at runtime. Before issue #52 this was a false negative: the
left-operand fallback echoed ``Decimal(10, 2)`` back and the mismatch
went unnoticed.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class DecIn(pa.DataFrameModel):
    d: pl.Decimal(10, 2)
    e: pl.Decimal(10, 2)


class StaleSum(pa.DataFrameModel):
    s: pl.Decimal(10, 2)

    class Config:
        strict = True


def add(df: DataFrame[DecIn]) -> DataFrame[StaleSum]:
    return df.select(s=pl.col("d") + pl.col("e"))
