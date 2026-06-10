"""Invalid: ``.bin`` namespace accessor on a non-Binary column (issue #51).

Probed on polars 1.41.2: ``pl.col("a").bin.encode("hex")`` with
``a: Int64`` raises ``SchemaError: invalid series dtype: expected
`Binary`, got `i64``` -> PLY012.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int


class Out(pa.DataFrameModel):
    hex: str


def encode(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(hex=pl.col("a").bin.encode("hex"))
