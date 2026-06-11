"""Invalid: validate() narrowing produces a specific type, not accept-anything.

False-negative twin of ``valid/pandera_validate_assign.py``: after
``CleanSchema.validate(raw)`` the variable is narrowed to CleanSchema
(``value: Float64``), so returning it where ``value: str`` is declared must
fail — this guards narrowing against degrading into an Unknown frame that
would satisfy any downstream declaration.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class RawSchema(pa.DataFrameModel):
    id: int


class CleanSchema(pa.DataFrameModel):
    id: int
    value: pl.Float64


class WrongOut(pa.DataFrameModel):
    id: int
    value: str  # WRONG: validate() narrowed value to Float64


def process(raw: DataFrame[RawSchema]) -> DataFrame[WrongOut]:
    df = CleanSchema.validate(raw)
    return df.select(pl.col("id"), pl.col("value"))
