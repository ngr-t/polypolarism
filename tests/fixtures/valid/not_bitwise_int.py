"""``not_()`` / ``~`` on integers is a dtype-preserving bitwise NOT (issue #72).

Documented contract (``Expr.not_`` docstring: "operates bitwise on
integers") and probed (polars 1.41.2): ``~Int64 -> Int64`` (``~1 == -2``);
a Boolean receiver stays Boolean. Before #72 ``not_``/``~`` were typed
Boolean unconditionally and the correct int declaration was falsely
rejected.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Ints(pa.DataFrameModel):
    a: int


class Flags(pa.DataFrameModel):
    flag: bool


class IntOut(pa.DataFrameModel):
    x: int


class BoolOut(pa.DataFrameModel):
    x: bool


def bitwise_not_keeps_int(df: DataFrame[Ints]) -> DataFrame[IntOut]:
    return df.select(x=pl.col("a").not_())


def invert_operator_keeps_int(df: DataFrame[Ints]) -> DataFrame[IntOut]:
    return df.select(x=~pl.col("a"))


def boolean_not_stays_boolean(df: DataFrame[Flags]) -> DataFrame[BoolOut]:
    return df.select(x=~pl.col("flag"))
