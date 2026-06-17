"""Invalid: rename mapping two sources onto one target (PLY015).

``df.rename({"a": "x", "b": "x"})`` maps both ``a`` and ``b`` to ``x`` —
producing two ``x`` columns. polars raises ``DuplicateError`` at runtime
(probed 1.41.2). The collision is provable from the mapping literal alone,
independent of frame contents.

Legal twin: ``valid/rename_no_collision``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    b: pl.Float64


class Out(pa.DataFrameModel):
    x: pl.Float64


def collide(df: DataFrame[In]) -> DataFrame[Out]:
    return df.rename({"a": "x", "b": "x"})
