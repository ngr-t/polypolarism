"""Invalid fixture: violations against object-API schemas (backlog C-11).

A strict ``pa.DataFrameSchema`` binds closed after ``validate`` — a
missing column is a provable pple-column-not-found — and declared dtypes drive the
expression rules (``.str`` on an int column is pple-wrong-namespace-dtype).
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


order_schema = pa.DataFrameSchema({"order_id": pa.Column(int)}, strict=True)


def missing_after_strict_validate(df: DataFrame[Src]) -> pl.DataFrame:
    out = order_schema.validate(df.select(order_id=pl.col("a")))
    return out.select(pl.col("missing"))


def str_on_int_column(df: DataFrame[Src]) -> pl.DataFrame:
    out = order_schema.validate(df.select(order_id=pl.col("a")))
    return out.select(pl.col("order_id").str.to_uppercase())
