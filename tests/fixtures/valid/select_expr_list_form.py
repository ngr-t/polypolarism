"""select([expr, ...]) list-of-Expr form is inferred, not dropped (issue #97).

Previously a list of expressions passed to ``select`` degraded to
"Could not infer return type"; the list elements are now flattened and
analyzed like the varargs form ``select(pl.col("k"), pl.col("v"))``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True
        coerce = False


@pa.check_types
def select_expr_list(df: DataFrame[KV]) -> DataFrame[KV]:
    return df.select([pl.col("k"), pl.col("v")])


@pa.check_types
def select_star_list(df: DataFrame[KV]) -> DataFrame[KV]:
    return df.select(*[pl.col("k"), pl.col("v")])
