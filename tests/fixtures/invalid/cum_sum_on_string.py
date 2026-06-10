"""cum_sum on a non-numeric column raises at runtime (issue #49, PLY016).

Probed (polars 1.41.2): ``pl.col(str).cum_sum()`` raises
``InvalidOperationError: `cum_sum` operation not supported for dtype `str```.
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
def bug_cum_sum_on_string(df: DataFrame[S]) -> DataFrame[Out]:
    return df.select(pl.col("s").cum_sum())
