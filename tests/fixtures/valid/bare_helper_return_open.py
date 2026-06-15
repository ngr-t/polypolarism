"""A bare ``-> pl.DataFrame`` helper's result binds OPEN at call sites (issue #103).

ADR-0006 binds bare frame *parameters* as open frames (assumption
semantics). For consistency, the *result* of calling a helper whose return
is a bare ``pl.DataFrame`` / ``pl.LazyFrame`` is also an open frame — not a
"could not infer return type" error. So a caller that returns such a result
under a ``DataFrame[Schema]`` annotation passes with leniency notes, exactly
like a bare-parameter source. The helpers below really produce ``{k, v, a}``,
so the assumed pass also holds at runtime.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class KV(pa.DataFrameModel):
    k: str
    v: float

    class Config:
        strict = True
        coerce = False


class KVa(pa.DataFrameModel):
    k: str
    v: float
    a: float

    class Config:
        strict = True
        coerce = False


def make_kva(df: DataFrame[KV]) -> pl.DataFrame:
    """Bare return annotation: opts in, claims no schema."""
    return df.with_columns(a=pl.col("v") * 2.0)


@pa.check_types
def h_direct(df: DataFrame[KV]) -> DataFrame[KVa]:
    return make_kva(df)


@pa.check_types
def h_reassign(df: DataFrame[KV]) -> DataFrame[KVa]:
    x = make_kva(df)
    return x
