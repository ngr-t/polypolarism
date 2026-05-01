"""explode + diagonal concat — pipeline that flattens lists then unions."""

from typing import Annotated

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class A(pa.DataFrameModel):
    user_id: int
    tags: Annotated[pl.List, pl.Utf8()]


class B(pa.DataFrameModel):
    user_id: int
    score: pl.Float64


def union(a: DataFrame[A], b: DataFrame[B]):
    return pl.concat([a.explode("tags"), b], how="diagonal")
