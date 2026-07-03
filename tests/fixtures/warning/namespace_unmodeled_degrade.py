"""Warning: unmodeled namespace methods honor the (name, Unknown) contract (issue #144).

An unmodeled method reached through an expression namespace (``.list``/``.struct``
/...) used to warn but then DISCARD the column (bare ``return None``), breaking
the ``(name, Unknown)`` degradation contract the direct-Expr path follows:

- FP: a positional select (``select(pl.col("vals").list.contains(3))``) hard-
  failed ``Could not infer return type`` even though the column name is knowable.
- FN: ``with_columns(pl.col("s").struct.json_encode())`` kept the stale precise
  ``Struct`` dtype (the namespace result was discarded), so ``s`` silently stayed
  wrong while the WARN text claimed it degraded.

Now the column registers as ``Unknown`` under its receiver name: the select
passes (against an open schema) and the with_columns overwrites ``s`` with
Unknown — both loud, both honest.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class ListIn(pa.DataFrameModel):
    vals: pl.List(pl.Int64)


class Open(pa.DataFrameModel):  # empty, non-strict
    pass


class StructIn(pa.DataFrameModel):
    s: pl.Struct({"a": pl.Int64, "b": pl.String})


def positional_namespace_select(df: DataFrame[ListIn]) -> DataFrame[Open]:
    return df.select(pl.col("vals").list.contains(3))


def stale_dtype_overwritten(df: DataFrame[StructIn]) -> DataFrame[StructIn]:
    return df.with_columns(pl.col("s").struct.json_encode())
