"""Invalid: ``pct_change`` on an int column declared int (issue #71).

False-negative twin of ``valid/pct_change_float.py``: ``pct_change``
divides, so the runtime dtype is Float64 — before #71 the shift-like
classification inferred Int64? and this wrong declaration passed
statically while pandera rejects the runtime Float64.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Ints(pa.DataFrameModel):
    a: int


class PctI64(pa.DataFrameModel):
    x: int = pa.Field(nullable=True)  # WRONG: pct_change yields Float64


def bug_int_declared(df: DataFrame[Ints]) -> DataFrame[PctI64]:
    return df.select(x=pl.col("a").pct_change())
