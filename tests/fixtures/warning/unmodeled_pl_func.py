"""Warning: unmodeled pl.* expression constructors loud-degrade (issue #140).

An unmodeled module-level ``pl.<fn>(...)`` (e.g. ``pl.fold`` / ``pl.reduce``) used
to degrade SILENTLY — no warning, and a wrong declared dtype passed. It now
degrades loudly with ``pplw-unmodeled-method`` (symmetric with an unmodeled
method): the result is Unknown, so the wrong declaration is no longer a silent
false negative — the warning tells the user to pin the dtype.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class XIn(pa.DataFrameModel):
    x: int


class Out(pa.DataFrameModel):
    r: int


def fold_unmodeled(df: DataFrame[XIn]) -> DataFrame[Out]:
    return df.select(r=pl.fold(0, lambda acc, v: acc + v, pl.col("x")))
