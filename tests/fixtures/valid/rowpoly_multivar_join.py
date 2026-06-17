"""Per-parameter row variables on a join helper (C-14 Tier 5).

``@rowpoly(a="R1", b="R2")`` names one row variable per frame parameter, so a
join helper preserves BOTH sides' caller extras. The body joins ``a`` and
``b`` on ``id``, which keeps every column of both — the preservation check
(distinct skolem per parameter) confirms neither R1 nor R2 is dropped, so the
decorator is accepted.

This fixture exercises recognition of the keyword form and the multi-variable
preservation check end to end. It is runtime-checkable: the harness
synthesizes ``a``/``b`` from A/B and the join validates against ``Joined`` —
the threading of *caller* extras only matters at call sites, which the unit
tests cover.

Invalid direction: ``invalid/rowpoly_drops_row_variable.py`` (a helper that
provably drops its row variable -> PLY043).
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame

from polypolarism import rowpoly


class A(pa.DataFrameModel):
    id: int

    class Config:
        strict = False


class B(pa.DataFrameModel):
    id: int
    tag: str

    class Config:
        strict = False


class Joined(pa.DataFrameModel):
    id: int
    tag: str

    class Config:
        strict = False


@rowpoly(a="R1", b="R2")
def merge(a: DataFrame[A], b: DataFrame[B]) -> DataFrame[Joined]:
    return a.join(b, on="id", how="inner")
