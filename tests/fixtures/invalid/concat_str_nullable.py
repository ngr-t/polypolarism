"""Invalid: concat_str under default ignore_nulls=False is nullable (issue #138).

``pl.concat_str(...)`` with the default ``ignore_nulls=False`` propagates nulls:
if any operand is null the whole result is null. With a nullable operand the
result is therefore nullable, so a non-null declaration is a false negative
(pandera rejects the null). ``ignore_nulls=True`` skips nulls and stays non-null
(valid/concat_str_ignore_nulls).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: str = pa.Field(nullable=True)
    b: str


class Out(pa.DataFrameModel):
    r: str  # non-null


def concat_str_default(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(r=pl.concat_str("a", "b"))
