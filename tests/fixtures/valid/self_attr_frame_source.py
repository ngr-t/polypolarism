"""Assignment from a frame-typed self.<attr> binds the target (issue #104).

An instance attribute stashed in __init__ from a frame-annotated parameter
resolves when a sibling method reads it — instead of being dropped (which
left the target at its stale prior type and hard-errored on later column
references). ``W.from_attr`` is the issue's false-positive repro; it now
matches its ``from_param`` control. ``Typed.passthrough`` shows a precise
``DataFrame[Schema]`` attribute flowing through exactly.
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


class W:
    def __init__(self, src: pl.DataFrame) -> None:
        self.src = src

    @pa.check_types
    def from_attr(self) -> DataFrame[KV]:
        acc = pl.DataFrame()
        acc = self.src
        return acc.sort("k").group_by("k").agg(pl.col("v").sum())

    @pa.check_types
    def from_attr_chain(self) -> DataFrame[KV]:
        acc = self.src.filter(pl.col("v") > 0)
        return acc.sort("k").group_by("k").agg(pl.col("v").sum())


class Typed:
    def __init__(self, df: DataFrame[KV]) -> None:
        self.df = df

    @pa.check_types
    def passthrough(self) -> DataFrame[KV]:
        x = self.df
        return x
