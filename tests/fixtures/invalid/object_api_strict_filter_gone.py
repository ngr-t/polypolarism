"""``strict="filter"`` provably removes undeclared columns (issue #88).

The validate output of a filter-mode schema is closed: a column the
schema does not declare is gone on every execution, so referencing it
is a provable missing-column error (PLY001-class, not the PLY042 lint —
the output never "admits extras at runtime").

False-positive twin: ``valid/object_api_strict_filter``.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    a: int
    b: str

    class Config:
        strict = True
        coerce = True


filter_schema = pa.DataFrameSchema({"a": pa.Column(pl.Int64)}, strict="filter")


def filtered_column_gone(df: DataFrame[Src]) -> pl.DataFrame:
    out = filter_schema.validate(df.select(pl.col("a"), pl.col("b")))
    return out.select(pl.col("b"))  # WRONG: filter-validate removed 'b'
