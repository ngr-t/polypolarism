"""Invalid: ``.cat`` namespace accessor on a non-Categorical/Enum column (issue #54).

Probed on polars 1.41.2: ``pl.col("a").cat.get_categories()`` with
``a: Int64`` raises ``SchemaError: invalid dtype: expected an Enum or
Categorical type, received 'Int64'`` — a plain String column is rejected
the same way -> pple-wrong-namespace-dtype.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    s: str


class Out(pa.DataFrameModel):
    cats: str


def cats_of_int(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(cats=pl.col("a").cat.get_categories())


def cats_of_str(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(cats=pl.col("s").cat.get_categories())
