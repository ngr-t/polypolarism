"""M4: unpivot produces fixed schema; hstack merges disjoint frames."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Wide(pa.DataFrameModel):
    id: int
    a: pl.Float64
    b: pl.Float64


class Side(pa.DataFrameModel):
    label: str


def long_with_label(wide: DataFrame[Wide], side: DataFrame[Side]):
    long = wide.unpivot(
        index=["id"],
        on=["a", "b"],
        variable_name="metric",
        value_name="amount",
    )
    return long.hstack(side)
