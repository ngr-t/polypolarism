"""A frame-typed self.<attr> is now precisely checked, not dropped (issue #104).

Because ``self.df`` (a ``DataFrame[KV]`` stashed in __init__) now resolves,
returning it where ``DataFrame[KVa]`` is declared is caught as a real
missing-column mismatch — rather than being dropped into a "could not infer"
(or, when seeded from an Unknown prior, silently passing: the false negative
the fix closes).
"""

from __future__ import annotations

import pandera.polars as pa
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


class Holder:
    def __init__(self, df: DataFrame[KV]) -> None:
        self.df = df

    @pa.check_types
    def wrong(self) -> DataFrame[KVa]:
        return self.df
