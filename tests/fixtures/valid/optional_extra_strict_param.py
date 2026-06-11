"""Valid fixture: Optional columns are not provable extras (issue #84).

A column declared ``Optional[T]`` (required=False) MAY be absent at
runtime — there are inputs on which passing the frame into a
``strict = True`` parameter succeeds, so the call site stays lenient.
Only required pins prove strict-extra violations (issue #82; see
``invalid/strict_param_extra_columns.py`` for the provable twin).
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class StrictPrice(pa.DataFrameModel):
    price: float

    class Config:
        strict = True
        coerce = True


class OptionalSku(pa.DataFrameModel):
    sku: str | None  # required=False: may be absent
    price: float

    class Config:
        strict = True
        coerce = True


@pa.check_types
def strict_helper(df: DataFrame[StrictPrice]) -> DataFrame[StrictPrice]:
    return df.filter(pl.col("price") > 0)


@pa.check_types
def pipeline(df: DataFrame[OptionalSku]) -> DataFrame[StrictPrice]:
    return strict_helper(df)
