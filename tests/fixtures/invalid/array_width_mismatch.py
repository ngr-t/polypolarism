"""`Array` width mismatches are dtype errors (backlog C-7, issue #53 gap).

False-negative twin of ``valid/array_dtype``: same passthrough operation,
only the declared width is wrong. Probed (polars 1.41.2): pandera rejects
a width mismatch at validation, and ``coerce`` cannot repair it — the
underlying cast raises "cannot cast Array to a different width".
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    q: pl.Array(pl.Int64, 3)


class Out(pa.DataFrameModel):
    q: pl.Array(pl.Int64, 5)  # WRONG: the column's width is provably 3


def passthrough(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select("q")
