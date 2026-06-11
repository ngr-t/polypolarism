"""hstack appends the side frame's columns with their own dtypes.

False-negative twin of ``valid/m4_unpivot_and_hstack.py`` (hstack side —
the unpivot side is already paired by ``unpivot_incompatible_values``).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Wide(pa.DataFrameModel):
    id: int
    a: pl.Float64
    b: pl.Float64


class Side(pa.DataFrameModel):
    label: str


class LongWrong(pa.DataFrameModel):
    id: int
    metric: str
    amount: pl.Float64
    label: pl.Int64  # WRONG: hstack appends Side.label as Utf8


def long_with_label(wide: DataFrame[Wide], side: DataFrame[Side]) -> DataFrame[LongWrong]:
    long = wide.unpivot(
        index=["id"],
        on=["a", "b"],
        variable_name="metric",
        value_name="amount",
    )
    return long.hstack(side)
