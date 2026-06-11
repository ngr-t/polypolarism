"""Valid fixture: validate-result bindings per strict mode (issue #88).

A strict=False validate passes the input's extras through — the result
binds as an open island, so a declared return needing those extras is
satisfiable (leniency, not MissingColumn). Coerce-repairable input
differences stay accepted (issue #89's leniency side).
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame

open_schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64)})


class Src(pa.DataFrameModel):
    a: int
    b: str

    class Config:
        strict = True


class NeedsB(pa.DataFrameModel):
    a: int
    b: str

    class Config:
        strict = False
        coerce = True


def open_obj_extras_survive(df: DataFrame[Src]) -> DataFrame[NeedsB]:
    out = open_schema.validate(df.select(pl.col("a"), pl.col("b")))
    return out
