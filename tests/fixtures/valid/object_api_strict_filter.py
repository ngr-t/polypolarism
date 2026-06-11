"""``strict="filter"`` schemas accept extra input columns (issue #88 boundary).

pandera's third strict mode: ``validate`` *removes* undeclared columns
instead of rejecting them, so feeding a wider frame is runtime-valid and
the result is exactly the declared columns.

False-negative twin: ``invalid/object_api_strict_filter_gone``.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    a: int
    b: str

    class Config:
        strict = True
        coerce = True


filter_schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64)}, strict="filter")


def filter_accepts_extras(df: DataFrame[Src]) -> pl.DataFrame:
    out = filter_schema.validate(df.select(pl.col("a"), pl.col("b")))
    return out.select(pl.col("a"))
