"""Valid fixture: bare ``pl.Struct`` is an OPEN struct (backlog C-9).

"Some struct, fields unknown" — probed: pandera validates any struct
against the bare declaration. Field lookups get assumption semantics
(``struct.field`` pins Unknown), and ``unnest`` opens the frame so the
unknown field columns stay referenceable.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class WithMeta(pa.DataFrameModel):
    id: int
    meta: pl.Struct

    class Config:
        strict = True


def field_lookup_is_assumed(df: pl.DataFrame) -> pl.DataFrame:
    out = WithMeta.validate(df)
    return out.select(pl.col("meta").struct.field("city"))


def unnest_opens_the_frame(df: pl.DataFrame) -> pl.DataFrame:
    out = WithMeta.validate(df)
    return out.unnest("meta").select(pl.col("id"), pl.col("anything"))
