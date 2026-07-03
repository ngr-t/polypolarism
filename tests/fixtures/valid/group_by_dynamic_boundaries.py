"""Valid: group_by_dynamic(include_boundaries=True) boundary columns (issues #141, #148).

``include_boundaries=True`` prepends ``_lower_boundary`` / ``_upper_boundary``
columns (the index dtype). Under a strict schema those underscore-named columns
must be declared via ``pa.Field(alias=...)`` — pandera ignores leading-underscore
attributes and validates under the alias (issue #148). This is the runtime-valid,
statically-checkable form: the boundary columns are added (#141) and matched by
alias under ``strict=True`` (#148).
"""

from datetime import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class TsIn(pa.DataFrameModel):
    ts: datetime
    v: int

    class Config:
        strict = True


class BoundOut(pa.DataFrameModel):
    lower: datetime = pa.Field(alias="_lower_boundary")
    upper: datetime = pa.Field(alias="_upper_boundary")
    ts: datetime
    s: int

    class Config:
        strict = True


def dynamic_boundaries(df: DataFrame[TsIn]) -> DataFrame[BoundOut]:
    return df.group_by_dynamic("ts", every="1d", include_boundaries=True).agg(s=pl.col("v").sum())
