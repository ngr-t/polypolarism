"""Invalid: semi/anti joins must keep the left schema exactly.

False-negative twin of ``valid/semi_anti_gather.py``: semi and anti joins
only filter rows — they add no right-side columns and change no dtypes.
A declaration that widens the schema with a right-side column, or changes
a left dtype, must fail.

Probed (polars 1.41): both join types return exactly the left schema
(``{'id': Int64, 'v': Int64}``); pandera rejects a required ``w`` column
(absent) and ``v: str`` (actual Int64) at runtime.
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class LeftSchema(pa.DataFrameModel):
    id: int
    v: int


class RightSchema(pa.DataFrameModel):
    id: int
    w: int


class SemiWidened(pa.DataFrameModel):
    id: int
    v: int
    w: int  # WRONG: a semi join adds no right-side columns


def semi_join_widened(
    left: DataFrame[LeftSchema],
    right: DataFrame[RightSchema],
) -> DataFrame[SemiWidened]:
    return left.join(right, on="id", how="semi")


class AntiWrongDtype(pa.DataFrameModel):
    id: int
    v: str  # WRONG: an anti join keeps the left dtype (Int64)


def anti_join_wrong_dtype(
    left: DataFrame[LeftSchema],
    right: DataFrame[RightSchema],
) -> DataFrame[AntiWrongDtype]:
    return left.join(right, on="id", how="anti")
