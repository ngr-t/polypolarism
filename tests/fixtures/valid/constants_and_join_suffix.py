"""Module-level constants resolve in column-spec args; join suffix= is honored.

End-to-end repro for issues #11 (custom join suffix) and #12 (column args
passed via module-level constants).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame

KEY = "id"
ON_COLS = ["a", "b"]


class Left(pa.DataFrameModel):
    id: int
    v: pl.Float64


class Right(pa.DataFrameModel):
    id: int
    v: pl.Float64


class Joined(pa.DataFrameModel):
    id: int
    v: pl.Float64
    v_new: pl.Float64


def join_via_constant_key(
    left: DataFrame[Left],
    right: DataFrame[Right],
) -> DataFrame[Joined]:
    """#12: on=KEY resolves; #11: suffix='_new' names the overlap v_new."""
    return left.join(right, on=KEY, how="inner", suffix="_new")


class Wide(pa.DataFrameModel):
    id: int
    a: pl.Float64
    b: pl.Float64


class Long(pa.DataFrameModel):
    id: int
    variable: str
    value: pl.Float64


def unpivot_via_constant(wide: DataFrame[Wide]) -> DataFrame[Long]:
    """#12: on=ON_COLS resolves instead of a false pple-unpivot."""
    return wide.unpivot(index=["id"], on=ON_COLS)
