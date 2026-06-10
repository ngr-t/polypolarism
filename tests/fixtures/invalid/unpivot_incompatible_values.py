"""Invalid test case: unpivot value columns have no common supertype (PLY022).

List + scalar has no polars supertype — unpivot raises at runtime
("'unpivot' not supported for dtype: list[i64]").
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class WideSchema(pa.DataFrameModel):
    id: int
    xs: pl.List(pl.Int64) = pa.Field()
    label: str


def unpivot_list_and_scalar(df: DataFrame[WideSchema]):
    """ERROR: List[Int64] and Utf8 cannot be unified into one value column."""
    return df.unpivot(index=["id"], on=["xs", "label"])
