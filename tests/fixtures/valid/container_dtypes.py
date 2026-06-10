"""Valid: container-typed schema fields register as columns (issue #10).

``pl.List(...)`` / ``pl.Array(...)`` / ``pl.Struct(...)`` call-form field
annotations resolve to columns with element types where possible.
Unnesting a struct whose fields are unknown yields an "open frame" that
may contain the declared output columns.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class OrderIn(pa.DataFrameModel):
    order_id: int
    items: pl.List(pl.Struct) = pa.Field()

    class Config:
        coerce = True


class LineOut(pa.DataFrameModel):
    order_id: int
    qty: int

    class Config:
        coerce = True


@pa.check_types
def explode_lines(df: DataFrame[OrderIn]) -> DataFrame[LineOut]:
    return df.explode("items").unnest("items")


class SeriesIn(pa.DataFrameModel):
    id: int
    vals: pl.List(pl.Int64) = pa.Field()
    q: pl.Array(pl.Int64, 4) = pa.Field()


class Exploded(pa.DataFrameModel):
    id: int
    vals: int


def explode_vals(df: DataFrame[SeriesIn]) -> DataFrame[Exploded]:
    return df.explode("vals")


class Totals(pa.DataFrameModel):
    id: int
    total: int


def array_total(df: DataFrame[SeriesIn]) -> DataFrame[Totals]:
    return df.select("id", total=pl.col("q").arr.sum())
