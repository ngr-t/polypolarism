"""An Optional frame-typed self.<attr> resolves to its frame type (issue #106).

Follow-up to #104: a ``self.<attr>`` stashed in __init__ from an Optional
frame parameter (``pl.DataFrame | None``, ``Optional[DataFrame[KV]]``) still
resolves to its non-None frame type, so ``acc = self.attr.select(...)`` inside
an ``is not None`` guard binds the target — instead of being dropped, which
left it at the empty ``pl.DataFrame()`` seed and hard-errored downstream.
"""

from __future__ import annotations

from typing import Optional

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
    def __init__(self, opt: pl.DataFrame | None) -> None:
        self.opt = opt

    @pa.check_types
    def optional_guarded(self) -> DataFrame[KV]:
        acc = pl.DataFrame()
        if self.opt is not None:
            acc = self.opt.select(["k", "v"])
        return acc.sort("k").group_by("k").agg(pl.col("v").sum())


class Typed:
    def __init__(self, opt: Optional[DataFrame[KV]]) -> None:  # noqa: UP045  (testing Optional[...] form)
        self.opt = opt

    @pa.check_types
    def guarded_return(self) -> DataFrame[KV]:
        if self.opt is not None:
            return self.opt
        return pl.DataFrame({"k": ["x"], "v": [1.0]})
