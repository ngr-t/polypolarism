"""cs.exclude + selector union/intersection/difference/complement."""

import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int
    price_a: pl.Float64
    price_b: pl.Float64
    name: str


def numeric_minus_id(df: DataFrame[S]):
    return df.select(cs.numeric() - cs.by_name("id"))


def prices_only(df: DataFrame[S]):
    return df.select(cs.numeric() & cs.starts_with("price_"))


def drop_strings(df: DataFrame[S]):
    return df.drop(cs.exclude(cs.numeric()))
