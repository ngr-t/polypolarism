"""`PLW008`: a variable annotation that NARROWS the inferred schema is an
unverified assertion — allowed (ADR-0005), but surfaced with a pointer to
``Schema.validate`` for a runtime-backed narrowing.

The left join makes ``segment`` nullable (`Int64?`); the annotation
asserts non-null ``Int64`` — the author knows every order id has a
customer. polypolarism cannot verify that, so the assertion warns.
"""

import pandera.polars as pa
import polars as pl  # noqa: F401  (fixtures are AST inputs; pl anchors the import convention)
from pandera.typing.polars import DataFrame


class Orders(pa.DataFrameModel):
    id: int


class Customers(pa.DataFrameModel):
    id: int
    segment: int


class Enriched(pa.DataFrameModel):
    id: int
    segment: int


def enrich(orders: DataFrame[Orders], customers: DataFrame[Customers]) -> DataFrame[Enriched]:
    joined: DataFrame[Enriched] = orders.join(customers, on="id", how="left")
    return joined


class SrcOpen(pa.DataFrameModel):
    a: int

    class Config:
        strict = False
        coerce = True


class WithB(pa.DataFrameModel):
    a: int
    b: str

    class Config:
        strict = False
        coerce = True


def narrow_open(df: DataFrame[SrcOpen]) -> DataFrame[SrcOpen]:
    # Issue #63: 'b' is not PROVABLY absent (the non-strict input schema
    # tolerates extra runtime columns) — narrowing, not PLY033.
    x: DataFrame[WithB] = df.filter(pl.col("a") > 0)  # noqa: F841
    return df


class StrOut(pa.DataFrameModel):
    a: str

    class Config:
        strict = True
        coerce = True


def coerce_unbacked(df: DataFrame[SrcOpen]) -> DataFrame[SrcOpen]:
    # Issue #64: the annotation relies on coerce=True, but annotations
    # never coerce at runtime — unbacked re-type, PLW008.
    y: DataFrame[StrOut] = df.select(a=pl.col("a"))  # noqa: F841
    return df
