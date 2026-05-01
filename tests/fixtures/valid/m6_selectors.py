"""polars.selectors (cs.*) expand to matching columns in select / drop."""

import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int
    price: pl.Float64
    name: str
    label: str


def numeric_only(df: DataFrame[S]):
    return df.select(cs.numeric())


def drop_strings(df: DataFrame[S]):
    return df.drop(cs.string())
