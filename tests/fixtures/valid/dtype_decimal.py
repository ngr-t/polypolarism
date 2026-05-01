"""Valid fixture: ``pl.Decimal(precision, scale)`` (polars 1.35+ stabilized).

Exercises landmark version 1.35 — Decimal as a stable, fixed-scale
dtype. Both bare ``pl.Decimal`` (default precision=38, scale=0) and the
parametrized ``pl.Decimal(p, s)`` form must round-trip preserving p / s.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class PriceSchema(pa.DataFrameModel):
    sku: int
    unit_price: pl.Decimal(20, 4)
    line_total: pl.Decimal(20, 4)


def passthrough(df: DataFrame[PriceSchema]) -> DataFrame[PriceSchema]:
    return df.filter(pl.col("sku") > 0)
