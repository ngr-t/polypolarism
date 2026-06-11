"""Valid: numeric-only elementwise methods on probed-valid receivers (issue #62).

``round``/``clip``/``abs``/``sign`` preserve the receiver dtype on every
probed-ACCEPTED cell (polars 1.41.2), including the surprising ones:
``round(1)`` on ``Decimal(10, 2)`` keeps precision AND scale, ``abs`` is
fine on ``Duration``, and ``sqrt`` returns Float64 from any numeric
receiver. False-positive twin of ``invalid/round_on_string.py``.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Measures(pa.DataFrameModel):
    price: float
    qty: int

    class Config:
        coerce = True


class Rounded(pa.DataFrameModel):
    price: float  # round keeps Float64
    qty: int  # abs/clip keep Int64
    root: float  # sqrt -> Float64

    class Config:
        strict = True
        coerce = True


@pa.check_types
def round_clip_abs(df: DataFrame[Measures]) -> DataFrame[Rounded]:
    return df.select(
        price=pl.col("price").round(1),
        qty=pl.col("qty").abs().clip(0, 100),
        root=pl.col("price").abs().sqrt(),
    )


class Ledger(pa.DataFrameModel):
    amount: pl.Decimal(10, 2)
    lag: pl.Duration

    class Config:
        coerce = True


class LedgerOut(pa.DataFrameModel):
    amount: pl.Decimal(10, 2)  # round(1) keeps precision AND scale (probed)
    lag: pl.Duration  # abs on Duration is dtype-preserving
    drift: pl.Decimal(10, 2)  # sign on Decimal keeps the dtype

    class Config:
        strict = True
        coerce = True


@pa.check_types
def decimal_duration_cells(df: DataFrame[Ledger]) -> DataFrame[LedgerOut]:
    return df.select(
        amount=pl.col("amount").round(1),
        lag=pl.col("lag").abs(),
        drift=pl.col("amount").sign(),
    )
