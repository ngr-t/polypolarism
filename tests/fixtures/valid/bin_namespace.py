"""Valid contrast for issue #51: ``.bin`` methods on a Binary column.

Probed return dtypes on polars 1.41.2: ``encode("hex")`` -> String,
``decode("hex")`` -> Binary, ``size()`` -> UInt32, ``starts_with`` ->
Boolean.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    payload: pl.Binary
    hex_repr: str


class Out(pa.DataFrameModel):
    hex: str
    decoded: bytes
    n_bytes: pl.UInt32
    is_png: bool


def inspect_payload(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        hex=pl.col("payload").bin.encode("hex"),
        decoded=pl.col("hex_repr").cast(pl.Binary).bin.decode("hex"),
        n_bytes=pl.col("payload").bin.size(),
        is_png=pl.col("payload").bin.starts_with(b"\x89PNG"),
    )
