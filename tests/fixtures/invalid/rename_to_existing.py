"""Invalid: rename whose target already exists in the frame (PLY015).

``df.rename({"a": "b"})`` where ``b`` is already a column — and ``b`` is not
itself renamed away — produces two ``b`` columns. polars raises
``DuplicateError`` at runtime (probed 1.41.2). The collision is provable
because ``b`` is a KNOWN present column.

Legal twin: ``valid/rename_no_collision`` (swap / fresh target stay clean).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    b: pl.Float64


class Out(pa.DataFrameModel):
    b: pl.Float64


def collide(df: DataFrame[In]) -> DataFrame[Out]:
    return df.rename({"a": "b"})
