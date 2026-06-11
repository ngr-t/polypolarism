"""Invalid: wrong declarations for coalesced / default-full / cross joins.

False-negative twin of ``valid/join_coalesce_cross.py``: each function uses
the same join as its valid counterpart but declares an impossible schema.

Probed (polars 1.41 + pandera):

- ``full + coalesce=True`` merges the keys: no ``id_right`` column exists,
  so declaring it fails at runtime (missing column).
- ``full`` WITHOUT coalesce keeps both keys and both are nullable (the
  probed result has nulls in ``id`` for right-only rows), so a
  non-nullable ``id`` fails at runtime.
- a cross join keeps right dtypes unchanged: ``label`` stays Utf8, so
  ``label: int`` fails at runtime.
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class Left(pa.DataFrameModel):
    id: int
    x: int


class Right(pa.DataFrameModel):
    id: int
    y: int


class FullCoalescedWrong(pa.DataFrameModel):
    id: int
    id_right: int = pa.Field(nullable=True)  # WRONG: coalesce=True merges the keys
    x: int = pa.Field(nullable=True)
    y: int = pa.Field(nullable=True)


def full_join_coalesced(
    left: DataFrame[Left],
    right: DataFrame[Right],
) -> DataFrame[FullCoalescedWrong]:
    return left.join(right, on="id", how="full", coalesce=True)


class FullDefaultWrong(pa.DataFrameModel):
    id: int  # WRONG: a non-coalesced full-join key is nullable
    x: int = pa.Field(nullable=True)
    id_right: int = pa.Field(nullable=True)
    y: int = pa.Field(nullable=True)


def full_join_default(
    left: DataFrame[Left],
    right: DataFrame[Right],
) -> DataFrame[FullDefaultWrong]:
    return left.join(right, on="id", how="full")


class RightFrame(pa.DataFrameModel):
    rid: int
    label: str


class CrossWrong(pa.DataFrameModel):
    id: int
    x: int
    rid: int
    label: int  # WRONG: cross join keeps the right dtype (Utf8)


def cross_join(
    left: DataFrame[Left],
    right: DataFrame[RightFrame],
) -> DataFrame[CrossWrong]:
    return left.join(right, how="cross")
