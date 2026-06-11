"""Invalid fixture: provable extra columns into a strict parameter (issue #82).

``check_types`` validates annotated ARGUMENTS too: a ``strict = True``
parameter schema rejects undeclared columns at runtime, so passing a
frame that provably carries extras is a call-site error — symmetrical
with the existing declared-return and annotated-assignment checks.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class StrictPrice(pa.DataFrameModel):
    price: float

    class Config:
        strict = True
        coerce = True


class Wide(pa.DataFrameModel):
    sku: str
    price: float

    class Config:
        strict = True
        coerce = True


@pa.check_types
def strict_helper(df: DataFrame[StrictPrice]) -> DataFrame[StrictPrice]:
    return df.filter(pl.col("price") > 0)


@pa.check_types
def wide_into_strict_param(df: DataFrame[Wide]) -> DataFrame[StrictPrice]:
    return strict_helper(df)  # closed {sku, price} into strict {price}
