"""Invalid: selector expansion feeding a wrong declared schema must fail.

False-negative twin of ``valid/m6_selectors.py`` / ``valid/m10_selector_algebra.py``:
``cs.numeric()`` expands to the numeric columns with their real dtypes, so a
declaration typing ``price`` as ``str`` — or expecting the string column
``name`` that the selector dropped — must be rejected. Guards selector
expansion against degrading to an accept-anything Unknown set.
"""

import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int
    price: pl.Float64
    name: str


class WrongDtypeOut(pa.DataFrameModel):
    id: int
    price: str  # WRONG: cs.numeric() keeps price as Float64


def numeric_only_wrong_dtype(df: DataFrame[S]) -> DataFrame[WrongDtypeOut]:
    return df.select(cs.numeric())


class DroppedColumnOut(pa.DataFrameModel):
    id: int
    name: str  # WRONG: cs.numeric() does not select the string column


def numeric_only_missing_column(df: DataFrame[S]) -> DataFrame[DroppedColumnOut]:
    return df.select(cs.numeric())
