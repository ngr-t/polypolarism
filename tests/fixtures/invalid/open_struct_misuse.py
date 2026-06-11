"""Invalid fixture: open structs keep their struct-ness (backlog C-9).

A bare ``pl.Struct`` column used to degrade to ``Unknown`` and silence
everything; it is provably a STRUCT (probed: ``.str`` on a struct column
is a runtime SchemaError), so wrong-namespace accessors are proofs.
Closed structs keep their exact field proofs.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class WithMeta(pa.DataFrameModel):
    id: int
    meta: pl.Struct

    class Config:
        strict = True


class WithAddr(pa.DataFrameModel):
    addr: pl.Struct({"city": pl.Utf8})

    class Config:
        strict = True


def str_on_open_struct(df: pl.DataFrame) -> pl.DataFrame:
    out = WithMeta.validate(df)
    return out.select(pl.col("meta").str.to_uppercase())


def field_typo_on_closed_struct(df: pl.DataFrame) -> pl.DataFrame:
    out = WithAddr.validate(df)
    return out.select(pl.col("addr").struct.field("ctiy"))
