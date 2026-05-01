"""Valid fixture: ``pl.Int128`` (polars 1.18+) as a Pandera schema dtype.

Exercises landmark version 1.18 — the ``Int128`` introduction. The
analyzer must recognize ``pl.Int128`` as a bare attribute and as a
``cast`` target, and must propagate it through identity-shape methods.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class LedgerSchema(pa.DataFrameModel):
    txn_id: int
    amount_micros: pl.Int128


def passthrough(df: DataFrame[LedgerSchema]) -> DataFrame[LedgerSchema]:
    return df.filter(pl.col("amount_micros") > 0)


def cast_to_int128(
    df: DataFrame[LedgerSchema],
) -> DataFrame[LedgerSchema]:
    return df.with_columns(pl.col("amount_micros").cast(pl.Int128))
