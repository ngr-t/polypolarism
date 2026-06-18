"""Invalid: ``.arr`` on a List column and ``.list`` on an Array column (issue #53).

Probed on polars 1.41.2: every ``.arr.*`` method on a List column raises
``InvalidOperationError: expected Array datatype for array operation,
got: List(Int64)`` and every ``.list.*`` method on an Array column raises
``expected List data type`` — the containers are not interchangeable.
Both directions -> pple-wrong-namespace-dtype.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    xs: pl.List(pl.Int64) = pa.Field()
    q: pl.Array(pl.Int64, 3) = pa.Field()


class Out(pa.DataFrameModel):
    total: int


def arr_on_list(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(total=pl.col("xs").arr.sum())


def list_on_array(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(total=pl.col("q").list.sum())
