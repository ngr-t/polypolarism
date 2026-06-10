"""Join key coalescing and cross joins.

End-to-end repros for issues #24 (full join with coalesce=True keeps the
key non-null) and #26 (how="cross" infers left columns + right columns).
Also pins the corrected full-join default: WITHOUT coalesce=True polars
keeps both key columns (the right one suffixed) and both are nullable.
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class Left(pa.DataFrameModel):
    id: int
    x: int


class Right(pa.DataFrameModel):
    id: int
    y: int


class FullCoalesced(pa.DataFrameModel):
    id: int  # coalesced key: null only if BOTH sides were null -> non-null
    x: int = pa.Field(nullable=True)
    y: int = pa.Field(nullable=True)

    class Config:
        strict = True


def full_join_coalesced(
    left: DataFrame[Left],
    right: DataFrame[Right],
) -> DataFrame[FullCoalesced]:
    """#24: coalesce=True merges the keys into one non-null column."""
    return left.join(right, on="id", how="full", coalesce=True)


class FullDefault(pa.DataFrameModel):
    id: int = pa.Field(nullable=True)
    x: int = pa.Field(nullable=True)
    id_right: int = pa.Field(nullable=True)
    y: int = pa.Field(nullable=True)

    class Config:
        strict = True


def full_join_default(
    left: DataFrame[Left],
    right: DataFrame[Right],
) -> DataFrame[FullDefault]:
    """Full joins do NOT coalesce by default: both keys survive, nullable."""
    return left.join(right, on="id", how="full")


class RightFrame(pa.DataFrameModel):
    rid: int
    label: str


class CrossOut(pa.DataFrameModel):
    id: int
    x: int
    rid: int
    label: str

    class Config:
        strict = True


def cross_join(
    left: DataFrame[Left],
    right: DataFrame[RightFrame],
) -> DataFrame[CrossOut]:
    """#26: cross join concatenates both schemas, no nullability added."""
    return left.join(right, how="cross")


class CrossCollision(pa.DataFrameModel):
    id: int
    x: int
    id_right: int
    y: int

    class Config:
        strict = True


def cross_join_collision(
    left: DataFrame[Left],
    right: DataFrame[Right],
) -> DataFrame[CrossCollision]:
    """Colliding right columns get the usual suffix in a cross join."""
    return left.join(right, how="cross")
