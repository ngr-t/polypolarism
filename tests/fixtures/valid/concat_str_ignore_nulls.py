"""Valid: concat_str stays non-null when nulls can't reach it (issue #138).

``ignore_nulls=True`` drops nulls before joining, so the result is non-null even
with a nullable operand; and concatenating only non-null operands is non-null
under the default. Both must type-check against a non-null declaration.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: str = pa.Field(nullable=True)
    b: str
    c: str


class Out(pa.DataFrameModel):
    r: str  # non-null


def concat_ignore_nulls(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(r=pl.concat_str("a", "b", ignore_nulls=True))


def concat_all_nonnull(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(r=pl.concat_str("b", "c"))
