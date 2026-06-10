"""Valid test case: schema-preserving semi/anti joins and gather_every (#15)."""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class LeftSchema(pa.DataFrameModel):
    id: int
    v: int

    class Config:
        coerce = True


class RightSchema(pa.DataFrameModel):
    id: int

    class Config:
        coerce = True


def ok_semi(
    left: DataFrame[LeftSchema],
    right: DataFrame[RightSchema],
) -> DataFrame[LeftSchema]:
    """Semi join keeps the left schema unchanged (just filters rows)."""
    return left.join(right, on="id", how="semi")


def ok_anti(
    left: DataFrame[LeftSchema],
    right: DataFrame[RightSchema],
) -> DataFrame[LeftSchema]:
    """Anti join keeps the left schema unchanged (just filters rows)."""
    return left.join(right, on="id", how="anti")


def ok_gather_every(left: DataFrame[LeftSchema]) -> DataFrame[LeftSchema]:
    """gather_every is schema-preserving (just fewer rows)."""
    return left.gather_every(2)
