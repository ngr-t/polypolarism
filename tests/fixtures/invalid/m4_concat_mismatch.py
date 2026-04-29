"""M4 invalid: vertical concat with mismatched column sets."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class A(pa.DataFrameModel):
    id: int


class B(pa.DataFrameModel):
    id: int
    extra: str


def f(a: DataFrame[A], b: DataFrame[B]):
    return pl.concat([a, b])
