"""pl.all()/pl.exclude() selectors, _seq variants, and constant column names.

End-to-end repro for issues #20 (pl.all / pl.exclude inside select),
#21 (select_seq / with_columns_seq dispatch), and #22 (column names
bound to module-level constants).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame

KEY = "a"


class TC(pa.DataFrameModel):
    a: int
    b: int
    name: str


class AB(pa.DataFrameModel):
    a: int
    b: int

    class Config:
        strict = True


class OutA(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


class WithSum(pa.DataFrameModel):
    a: int
    b: int
    total: int


def sel_all(df: DataFrame[TC]) -> DataFrame[TC]:
    """#20: pl.all() selects every input column."""
    return df.select(pl.all())


def sel_exclude(df: DataFrame[TC]) -> DataFrame[AB]:
    """#20: pl.exclude(names) selects everything but the named columns."""
    return df.select(pl.exclude("name"))


def seq_pipeline(df: DataFrame[AB]) -> DataFrame[WithSum]:
    """#21: the _seq variants infer like select / with_columns."""
    return df.with_columns_seq(total=pl.col("a") + pl.col("b")).select_seq("a", "b", "total")


def select_via_constant(df: DataFrame[AB]) -> DataFrame[OutA]:
    """#22: a module-level constant resolves as a column name."""
    return df.select(KEY)


def select_seq_via_constant(df: DataFrame[AB]) -> DataFrame[OutA]:
    """#21 x #22: constants resolve through the _seq variant too."""
    return df.select_seq(KEY)
