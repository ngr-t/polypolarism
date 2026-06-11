"""struct.rename_fields renames positionally — the dtypes are unchanged.

False-negative twin of ``valid/struct_rename_fields.py``. The output
schema deliberately drops the valid twin's ``coerce = True``: with coerce,
the Int64 -> Utf8 difference is tolerated (anything formats to String),
which would mask exactly the regression this twin exists to catch.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class SX(pa.DataFrameModel):
    s: pl.Struct({"x": pl.Int64, "y": pl.Int64})

    class Config:
        coerce = True


class PQ(pa.DataFrameModel):
    p: int
    q: str  # WRONG: rename_fields only renames; 'q' is the old Int64 'y'

    class Config:
        strict = True


@pa.check_types
def bad_struct_rename(df: DataFrame[SX]) -> DataFrame[PQ]:
    return df.select(pl.col("s").struct.rename_fields(["p", "q"])).unnest("s")
