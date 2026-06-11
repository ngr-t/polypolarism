"""Frame-literal schema inference vs wrong declarations.

False-negative twin of ``valid/frame_literal.py``: the dict-literal and
explicit ``schema=`` constructor forms, each with a wrong declared dtype.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Empty(pa.DataFrameModel):
    pass


class Lit(pa.DataFrameModel):
    a: str  # WRONG: the literal list [1, 2, 3] infers Int64

    class Config:
        strict = True


@pa.check_types
def pure_literal(df: DataFrame[Empty]) -> DataFrame[Lit]:
    return pl.DataFrame({"a": [1, 2, 3]})


class Typed(pa.DataFrameModel):
    a: pl.Int64  # WRONG: the explicit schema= pins 'a' to Int32
    b: pl.Int8


@pa.check_types
def explicit_schema(df: DataFrame[Empty]) -> DataFrame[Typed]:
    return pl.DataFrame(
        {"a": [1], "b": [2]},
        schema={"a": pl.Int32, "b": pl.Int8},
    )
