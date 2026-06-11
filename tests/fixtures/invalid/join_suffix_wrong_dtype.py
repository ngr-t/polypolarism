"""Invalid: wrong dtype declared for a suffixed join-collision column.

False-negative twin of ``valid/constants_and_join_suffix.py``: the inner
join with ``suffix="_new"`` names the colliding right-side ``v`` column
``v_new`` and its dtype stays Float64, so declaring ``v_new: int`` must
fail — this guards the suffix rule against degrading into an Unknown
column that would satisfy any declaration.

Probed (polars 1.41): the joined schema is ``{'id': Int64, 'v': Float64,
'v_new': Float64}`` and pandera rejects ``v_new: int`` at runtime.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Left(pa.DataFrameModel):
    id: int
    v: pl.Float64


class Right(pa.DataFrameModel):
    id: int
    v: pl.Float64


class Joined(pa.DataFrameModel):
    id: int
    v: pl.Float64
    v_new: int  # WRONG: the suffixed right-side column keeps Float64


def join_with_suffix(
    left: DataFrame[Left],
    right: DataFrame[Right],
) -> DataFrame[Joined]:
    return left.join(right, on="id", how="inner", suffix="_new")
