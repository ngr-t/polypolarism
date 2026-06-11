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
