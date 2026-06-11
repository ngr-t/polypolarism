"""Valid fixture: implicit open-frame sources (ADR-0006, backlog C-12a).

A bare ``pl.DataFrame`` / ``pl.LazyFrame`` parameter and the ``pl.read_*``
/ ``pl.scan_*`` readers bind an empty OPEN frame: nothing is known about
the source's columns, but everything the body itself determines is
checked. Column lookups on an open frame are ASSUMED to succeed (their
absence is never provable — the gradual-typing boundary); shape-
determining calls (``select``) close the frame, making everything
downstream fully checkable; a ``DataFrame[Schema]`` return checks the
pinned columns exactly and records the unverifiable rest as leniency.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


def bare_helper_pins_what_it_builds(df: pl.DataFrame) -> pl.DataFrame:
    # ``ratio`` is pinned (Float64); the input columns stay unknown.
    return df.with_columns(ratio=pl.col("num") / pl.col("den"))


def scan_select_closes_the_frame(path: str = "data.parquet") -> pl.DataFrame:
    # ``select`` output shape is determined by the call itself — the frame
    # is closed downstream even though the parquet schema is unknown.
    return pl.scan_parquet(path).select("a", "b").collect()


class Report(pa.DataFrameModel):
    total: float
    label: str

    class Config:
        strict = True


def open_frame_to_declared_schema(df: pl.DataFrame) -> DataFrame[Report]:
    # Both declared columns are pinned by the body with exact dtypes —
    # the declared return is checked precisely despite the open source.
    out = df.select(
        total=pl.col("amount").cast(pl.Float64),
        label=pl.col("name").cast(pl.Utf8),
    )
    return Report.validate(out)
