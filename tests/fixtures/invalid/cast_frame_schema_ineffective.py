"""typing.cast does not override the real schema; the mismatch still FAILs (issue #102).

The cast claims ``DataFrame[KVa]`` but the argument's real schema is ``{k}`` —
polypolarism checks the real schema, so pple-return-type fires (the cast is inert,
exactly as it is at runtime under ``@pa.check_types``). pplw-ignored-cast accompanies the
error to explain that ``typing.cast`` is not honored as an assertion and that
``# type: ignore[pple-return-type]`` is the way to suppress it if intentional.
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
def cast_ineffective(df: DataFrame[KV]) -> DataFrame[KVa]:
    return cast(DataFrame[KVa], df.select(pl.col("k")))
