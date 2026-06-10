"""struct.rename_fields applies new names positionally (issue #48).

Probed (polars 1.41.2): ``rename_fields(["p", "q"])`` zips the new names
onto the existing fields in order; fewer names truncate the struct to the
renamed prefix without error.
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
    q: int

    class Config:
        strict = True
        coerce = True


@pa.check_types
def ok_struct_rename_fields(df: DataFrame[SX]) -> DataFrame[PQ]:
    return df.select(pl.col("s").struct.rename_fields(["p", "q"])).unnest("s")


class POnly(pa.DataFrameModel):
    p: int

    class Config:
        strict = True
        coerce = True


@pa.check_types
def ok_fewer_names_truncate(df: DataFrame[SX]) -> DataFrame[POnly]:
    # Probed: polars drops the un-named trailing fields without error.
    return df.select(pl.col("s").struct.rename_fields(["p"])).unnest("s")
