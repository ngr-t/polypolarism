"""Invalid: arithmetic between incompatible dtypes (issue #30).

``pl.col("s") + pl.col("n")`` adds a String to an Int64 — polars raises
``InvalidOperationError`` at runtime ("arithmetic on string and numeric
not allowed"). polypolarism flags it statically as pple-incompatible-operands.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    s: str
    n: int


class Out(pa.DataFrameModel):
    r: int


def combine(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(r=pl.col("s") + pl.col("n"))
