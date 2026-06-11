"""Invalid: ``map_elements(..., return_dtype=)`` contradicting the declaration.

False-negative twin of ``valid/m7_map_elements_dtype.py``: ``return_dtype=``
is the user's own dtype pin, so a declared column dtype that contradicts it
must fail — this proves the pin actually feeds the check instead of being a
decorative annotation.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    id: int
    value: pl.Float64


class Out(pa.DataFrameModel):
    id: int
    value: pl.Float64
    doubled: str  # WRONG: return_dtype pins Float64


def f(df: DataFrame[In]) -> DataFrame[Out]:
    return df.with_columns(
        pl.col("value").map_elements(lambda v: v * 2.0, return_dtype=pl.Float64).alias("doubled")
    )
