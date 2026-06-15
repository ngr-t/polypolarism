"""typing.cast to a frame schema emits PLW013 (verdict unchanged) (issue #102).

polypolarism does not honor ``typing.cast(DataFrame[Schema], x)`` as a schema
assertion — it infers ``x``'s real schema and checks that. The cast is silent
otherwise, which surprises users from mypy (where cast silences the checker),
so PLW013 makes it visible. These functions all PASS; the note is advisory:

- ``cast_match`` — the cast argument's real schema matches the declared
  return, so the function is OK; the note flags that the cast was inert.
- ``cast_unverified`` / ``cast_unverified_via_var`` — the source is an open
  ``pl.DataFrame``, so the assertion is accepted but unverifiable.
"""

from __future__ import annotations

from typing import cast

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


@pa.check_types
def cast_match(df: DataFrame[KV]) -> DataFrame[KVa]:
    return cast(DataFrame[KVa], df.with_columns(a=pl.col("v") * 2.0))


@pa.check_types
def cast_unverified(raw: pl.DataFrame) -> DataFrame[KVa]:
    return cast(DataFrame[KVa], raw)


@pa.check_types
def cast_unverified_via_var(raw: pl.DataFrame) -> DataFrame[KVa]:
    x = cast(DataFrame[KVa], raw)
    return x
