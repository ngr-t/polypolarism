"""Valid: hstack([pl.Series(name, ..., dtype=T)]) contributes the column (issue #134).

The documented list-of-Series form of ``hstack`` adds columns from Series
literals. A ``pl.Series("q3", ..., dtype=pl.Int64)`` carries a statically
knowable name and dtype, so the result schema includes ``q3`` and the declared
return must type-check (previously the hstack added nothing silently).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Wide(pa.DataFrameModel):
    id: str
    q1: int


class WideQ3(pa.DataFrameModel):
    id: str
    q1: int
    q3: int


def hstack_series(df: DataFrame[Wide]) -> DataFrame[WideQ3]:
    return df.hstack([pl.Series("q3", [0] * df.height, dtype=pl.Int64)])
