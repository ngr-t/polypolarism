"""Valid fixture: pandera object-API schemas (backlog C-11, tiers 1-2).

Module-level ``pa.DataFrameSchema({...})`` assignments register like
class schemas, keyed by the variable name — ``schema.validate(df)``
narrowing, strict closure and dtype checking all apply. Tier-2
construction folds statically: dict comprehensions over string-list
constants, and ``add_columns`` derivation.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


order_schema = pa.DataFrameSchema(
    {"order_id": pa.Column(int), "amount": pa.Column(float)},
    strict=True,
)

METRICS = ["m1", "m2"]

metrics_schema = pa.DataFrameSchema({c: pa.Column(float) for c in METRICS}, strict=True)

labeled_schema = metrics_schema.add_columns({"label": pa.Column(str)})


def validate_narrows(df: DataFrame[Src]) -> pl.DataFrame:
    out = order_schema.validate(
        df.select(order_id=pl.col("a"), amount=pl.col("a").cast(pl.Float64))
    )
    return out.select(pl.col("amount") * 2)


def tier2_construction(df: DataFrame[Src]) -> pl.DataFrame:
    built = df.select(
        m1=pl.col("a").cast(pl.Float64),
        m2=pl.col("a").cast(pl.Float64),
        label=pl.lit("x"),
    )
    out = labeled_schema.validate(built)
    return out.select(pl.col("m1") + pl.col("m2"), pl.col("label"))
