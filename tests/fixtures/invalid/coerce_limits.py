"""Invalid: ``Config.coerce`` has limits — it is not accept-anything.

False-negative twin of ``valid/coerce_len_agg.py``: coerce tolerates
numeric dtype differences (UInt32 -> Int64), but it must NOT excuse

- a non-numeric mismatch: ``label`` is built with ``pl.lit("total")``
  (Utf8); declaring it ``int`` fails at runtime because pandera's coerce
  cast of the string "total" to Int64 raises, and
- nullability: ``shift(1)`` puts a null in the first row; coercion casts
  dtypes but does not remove nulls, so a non-nullable ``prev`` fails.

Probed (polars 1.41 + pandera): both validations raise SchemaError.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class SalesSchema(pa.DataFrameModel):
    country: str
    amount: int

    class Config:
        coerce = True


class CountsWrong(pa.DataFrameModel):
    country: str
    n: int  # pl.len() yields UInt32 -> Int64: genuinely coercible
    label: int  # WRONG: Utf8 -> Int64 is not coercible, even under coerce

    class Config:
        coerce = True


def counts_with_label(sales: DataFrame[SalesSchema]) -> DataFrame[CountsWrong]:
    return sales.group_by("country").agg(n=pl.len(), label=pl.lit("total"))


class ShiftWrong(pa.DataFrameModel):
    country: str
    amount: int
    prev: int  # WRONG: shift(1) makes the column nullable; coerce keeps nulls

    class Config:
        coerce = True


def shifted(sales: DataFrame[SalesSchema]) -> DataFrame[ShiftWrong]:
    return sales.with_columns(prev=pl.col("amount").shift(1))
