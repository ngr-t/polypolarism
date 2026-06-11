"""join_where degrades to an open frame with PLW007 (issue #74).

polars marks ``join_where`` experimental ("It may be changed at any point
without it being considered a breaking change"), so polypolarism
deliberately does NOT encode its schema: the result is an open frame, the
declared return passes via the open-frame leniency (the ``via:`` notes in
the golden), and PLW007 keeps the degradation visible. The declared
schema below matches the observed polars output (left + right columns,
``_right`` suffix on collisions; probed identical on 1.37.0-1.41.2) — a
candidate for precise inference if/when polars stabilizes the API.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class LeftK(pa.DataFrameModel):
    k: int
    x: int


class RightK(pa.DataFrameModel):
    k: int
    y: int


class JoinedWhere(pa.DataFrameModel):
    k: int
    x: int
    k_right: int
    y: int

    class Config:
        strict = True


def ok_join_where(a: DataFrame[LeftK], b: DataFrame[RightK]) -> DataFrame[JoinedWhere]:
    """The issue #74 repro: correct code no longer hard-fails."""
    return a.join_where(b, pl.col("x") < pl.col("y"))
