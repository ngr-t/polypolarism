"""Invalid: ``not_()`` / ``~`` misused on int and float columns (issue #72).

False-negative twin of ``valid/not_bitwise_int.py``:

- on an int column the runtime result is a bitwise NOT (Int64), so a
  Boolean declaration fails pandera validation at runtime — before #72
  the unconditional Boolean inference let it pass statically;
- on a float column polars raises ``InvalidOperationError: dtype Float64
  not supported in 'not' operation`` (probed, 1.41.2) -> pple-non-numeric-operand.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Ints(pa.DataFrameModel):
    a: int


class Floats(pa.DataFrameModel):
    f: pl.Float64


class BoolOut(pa.DataFrameModel):
    x: bool  # WRONG: ~int is a bitwise NOT, not a Boolean


class FloatOut(pa.DataFrameModel):
    x: pl.Float64


def bug_bool_declared(df: DataFrame[Ints]) -> DataFrame[BoolOut]:
    return df.select(x=~pl.col("a"))


def bug_not_on_float(df: DataFrame[Floats]) -> DataFrame[FloatOut]:
    return df.select(x=pl.col("f").not_())
