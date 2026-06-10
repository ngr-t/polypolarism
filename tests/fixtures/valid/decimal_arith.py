"""Valid fixture: Decimal arithmetic propagates precision growth (issue #52).

Probed on polars 1.41.2: ``Decimal(p1,s1) <op> Decimal(p2,s2)`` saturates
the precision to 38 for ``+ - * /`` with scale ``max(s1, s2)`` (``*`` does
NOT add scales and ``/`` stays Decimal); ``Decimal <op> int`` widens to
``Decimal(38, scale)`` for every integer width; ``Decimal <op> float``
yields Float64. Declaring the grown dtypes must pass — the old
left-operand fallback claimed a stale ``Decimal(10, 2)`` and produced a
false positive against ``Decimal(38, 2)`` declarations.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class DecIn(pa.DataFrameModel):
    d: pl.Decimal(10, 2)
    e: pl.Decimal(10, 2)
    fine: pl.Decimal(12, 4)
    qty: int
    rate: float

    class Config:
        coerce = True


class DecSum(pa.DataFrameModel):
    """The issue #52 repro: Decimal + Decimal -> Decimal(38, 2)."""

    s: pl.Decimal(38, 2)

    class Config:
        strict = True
        coerce = True


@pa.check_types
def add(df: DataFrame[DecIn]) -> DataFrame[DecSum]:
    return df.select(s=pl.col("d") + pl.col("e"))


class DecGrown(pa.DataFrameModel):
    """Mixed partners: int keeps the scale, mixed scales take the max,
    a float partner leaves the Decimal domain entirely."""

    total: pl.Decimal(38, 2)
    ratio: pl.Decimal(38, 4)
    scaled: float

    class Config:
        strict = True
        coerce = True


@pa.check_types
def derive(df: DataFrame[DecIn]) -> DataFrame[DecGrown]:
    return df.select(
        total=pl.col("d") * pl.col("qty"),
        ratio=pl.col("d") / pl.col("fine"),
        scaled=pl.col("e") * pl.col("rate"),
    )
