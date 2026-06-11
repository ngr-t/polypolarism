"""Invalid fixture for issue #69 (PLY040), import-time variant.

Single-argument ``Annotated[pl.Datetime]`` is rejected by Python's typing
module itself ("Annotated[...] should be used with at least two arguments"),
so this module crashes at IMPORT time — before pandera ever sees the class.
polypolarism never imports the checked file, so flagging the annotation is
the only way the user finds out before running it.

Runtime differential: whole-fixture SKIP (the module cannot be imported at
all; the import-time crash IS the diagnostic's subject).
"""

from __future__ import annotations

import typing
from datetime import datetime

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    t: datetime

    class Config:
        strict = True


class NoMetadata(pa.DataFrameModel):
    t: typing.Annotated[pl.Datetime]  # TypeError at import: needs >= 2 args

    class Config:
        strict = True


def uses_no_metadata_schema(df: DataFrame[Src]) -> DataFrame[NoMetadata]:
    return df.select("t")
