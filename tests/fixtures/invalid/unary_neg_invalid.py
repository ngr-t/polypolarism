"""Invalid: unary minus on an unsigned / Boolean receiver (issue #136).

``-pl.col(...)`` (``ast.USub``) raises ``InvalidOperationError`` at runtime for
unsigned integers and Boolean (and Date/Datetime/Time). It was silently
unmodeled — the invalid receiver passed with no diagnostic. Now it is flagged
``pple-non-numeric-operand``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    u32: pl.UInt32
    b: bool


class Out(pa.DataFrameModel):
    r: int


def neg_unsigned(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(r=-pl.col("u32"))


def neg_boolean(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(r=-pl.col("b"))
