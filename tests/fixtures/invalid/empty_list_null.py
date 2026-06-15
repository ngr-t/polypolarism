"""pl.DataFrame({"c": []}) columns are Null, not Unknown — caught under coerce=False (issue #101)."""

from __future__ import annotations

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class KVi(pa.DataFrameModel):
    k: str
    v: pl.Int64

    class Config:
        strict = True
        coerce = False


def fn_empty_lists() -> DataFrame[KVi]:
    """Empty-list columns are Null; Null != Int64 under coerce=False."""
    return pl.DataFrame({"k": [], "v": []})
