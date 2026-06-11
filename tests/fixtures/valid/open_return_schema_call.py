"""Valid fixture: non-strict return schemas bind OPEN at call sites (issue #81).

``strict = False`` is pandera's "at least these columns" — the helper
preserves the caller's extra columns and ``check_types`` passes them
through, so the call result is an open frame: the caller can keep using
its own columns (``sku`` resolves through the rest as Unknown — the
pandera-expressible signature can't share the row variable between
input and output, so leniency, not a closed-frame error).
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class HasPrice(pa.DataFrameModel):
    price: float

    class Config:
        strict = False
        coerce = True


class HasPriceTotal(pa.DataFrameModel):
    price: float
    total: float

    class Config:
        strict = False
        coerce = True


class WideSales(pa.DataFrameModel):
    sku: str
    price: float

    class Config:
        strict = True
        coerce = True


class WideOut(pa.DataFrameModel):
    sku: str
    price: float
    total: float

    class Config:
        strict = True
        coerce = True


@pa.check_types
def add_total(df: DataFrame[HasPrice]) -> DataFrame[HasPriceTotal]:
    return df.with_columns(total=pl.col("price") * 1.1)


@pa.check_types
def pipeline(df: DataFrame[WideSales]) -> DataFrame[WideOut]:
    out = add_total(df)
    return out.select(pl.col("sku"), pl.col("price"), pl.col("total"))
