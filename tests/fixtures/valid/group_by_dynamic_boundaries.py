"""Valid: group_by_dynamic(include_boundaries=True) adds boundary columns (issue #141).

``include_boundaries=True`` prepends ``_lower_boundary`` / ``_upper_boundary``
columns (the index column's dtype, non-null) to the output. The keyword was not
read, so declaring them was rejected as missing columns.
"""

from datetime import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class TsIn(pa.DataFrameModel):
    ts: datetime
    v: int


class BoundOut(pa.DataFrameModel):
    _lower_boundary: datetime
    _upper_boundary: datetime
    ts: datetime
    s: int


def dynamic_boundaries(df: DataFrame[TsIn]) -> DataFrame[BoundOut]:
    return df.group_by_dynamic("ts", every="1d", include_boundaries=True).agg(s=pl.col("v").sum())
