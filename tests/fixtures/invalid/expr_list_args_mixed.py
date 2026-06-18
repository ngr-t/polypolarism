"""Invalid: list literal mixed with other positional args (pple-list-literal-misuse, issue #59).

False-negative twin of ``valid/expr_list_args.py``: the multi-expression
helpers flatten EITHER varargs OR one single list — a mix is never
flattened at runtime. Probed (polars 1.41.2): every function below raises
TypeError ("not yet implemented: Nested object types") when called, so the
pre-#59 flatten-as-varargs typing silently accepted crashing code.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    b: int


class StructOut(pa.DataFrameModel):
    a: int
    b: int


def struct_mixed_args(df: DataFrame[In]) -> DataFrame[StructOut]:
    # WRONG: the issue #59 repro — the list parses as a nested literal,
    # polars raises TypeError instead of building Struct{a, b}.
    return df.select(s=pl.struct("a", [pl.col("b")])).unnest("s")


class CoOut(pa.DataFrameModel):
    c: int


def coalesce_mixed_args(df: DataFrame[In]) -> DataFrame[CoOut]:
    # WRONG: same mix on pl.coalesce — TypeError at runtime.
    return df.select(c=pl.coalesce(pl.col("a"), [pl.col("b")]))


class TotalOut(pa.DataFrameModel):
    total: int


def horizontal_mixed_args(df: DataFrame[In]) -> DataFrame[TotalOut]:
    # WRONG: same mix on pl.sum_horizontal — TypeError at runtime.
    return df.select(total=pl.sum_horizontal([pl.col("a")], pl.col("b")))
