"""pl.concat_str / pl.coalesce constructor dtypes vs wrong declarations.

False-negative twin of ``valid/m6_pl_constructors.py``. The ``pl.struct``
leg of the valid fixture already has its own invalid twin
(``struct_field_wrong_dtype``), so this file pairs the remaining two
constructors.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    first: str
    last: str
    primary: pl.Float64 = pa.Field(nullable=True)
    fallback: pl.Float64


class RecNameWrong(pa.DataFrameModel):
    full_name: int  # WRONG: concat_str always yields Utf8
    amount: pl.Float64


def concat_str_wrong(df: DataFrame[S]) -> DataFrame[RecNameWrong]:
    return df.select(
        pl.concat_str([pl.col("first"), pl.col("last")], separator=" ").alias("full_name"),
        pl.coalesce(pl.col("primary"), pl.col("fallback")).alias("amount"),
    )


class RecAmountWrong(pa.DataFrameModel):
    full_name: str
    amount: pl.Int64  # WRONG: coalesce of Float64 columns stays Float64


def coalesce_wrong(df: DataFrame[S]) -> DataFrame[RecAmountWrong]:
    return df.select(
        pl.concat_str([pl.col("first"), pl.col("last")], separator=" ").alias("full_name"),
        pl.coalesce(pl.col("primary"), pl.col("fallback")).alias("amount"),
    )
