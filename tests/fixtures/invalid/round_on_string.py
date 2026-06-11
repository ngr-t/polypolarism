"""round on a non-numeric column raises at runtime (issue #62, PLY016).

Probed (polars 1.41.2): ``pl.col(str).round(1)`` raises
``InvalidOperationError: rounding ('half_to_even') can only be used on
numeric types``. False-negative twin of ``valid/numeric_elementwise.py``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    s: str

    class Config:
        coerce = True


class Out(pa.DataFrameModel):
    s: str

    class Config:
        coerce = True


@pa.check_types
def bug_round_on_string(df: DataFrame[S]) -> DataFrame[Out]:
    return df.select(pl.col("s").round(1))  # WRONG: round is numeric-only (PLY016)
