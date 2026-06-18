"""Valid fixture: stdlib ``decimal.Decimal`` / ``datetime.time`` annotations.

Issue #77: these two stdlib classes were missing from the alias maps, so the
fields silently vanished from the schema — correct code was rejected against
strict declarations (every returned column looked "extra"). Pandera resolves
both (probed, 0.31.1): ``decimal.Decimal`` -> ``Decimal(28, 0)`` (the same
engine default as a bare ``pl.Decimal``, issue #75) and ``datetime.time`` ->
``Time``; ``validate`` passes the exactly-matching frames below. Both the
qualified (``decimal.Decimal`` / ``datetime.time``) and bare-import
(``from decimal import Decimal`` / ``from datetime import time``) spellings
are exercised.
"""

from __future__ import annotations

import datetime
import decimal
from datetime import time
from decimal import Decimal

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    a: int

    class Config:
        strict = True


class DecStrict(pa.DataFrameModel):
    d: decimal.Decimal  # pandera: Decimal(28, 0)

    class Config:
        strict = True


class TimeStrict(pa.DataFrameModel):
    t: datetime.time  # pandera: Time

    class Config:
        strict = True


class BareNames(pa.DataFrameModel):
    d: Decimal  # from decimal import Decimal -> Decimal(28, 0)
    t: time  # from datetime import time -> Time

    class Config:
        strict = True


def qualified_decimal(df: DataFrame[Src]) -> DataFrame[DecStrict]:
    # The issue #77 FP repro: this was "[pple-return-type] Extra column 'd'".
    return df.select(d=pl.col("a").cast(pl.Decimal(28, 0)))


def qualified_time(df: DataFrame[Src]) -> DataFrame[TimeStrict]:
    return df.select(t=pl.col("a").cast(pl.Time))


def bare_imports(df: DataFrame[Src]) -> DataFrame[BareNames]:
    return df.select(
        d=pl.col("a").cast(pl.Decimal(28, 0)),
        t=pl.col("a").cast(pl.Time),
    )
