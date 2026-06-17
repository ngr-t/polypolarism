"""Valid: renames / projections that do NOT produce a duplicate column.

False-positive twins of ``invalid/rename_to_existing`` and
``invalid/rename_duplicate_target``. Each case runs cleanly on polars 1.41.2:

* a simultaneous swap renames the colliding name away as the value moves in;
* a rename to a fresh name never collides;
* ``with_columns`` REPLACING an existing column is legal (overwrite, not
  duplicate);
* a ``select`` whose outputs all have distinct names is fine.

polypolarism must keep all of these clean — the duplicate-output check only
fires on a PROVABLE collision.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    a: int
    b: pl.Float64


def rename_swap(df: DataFrame[In]) -> pl.DataFrame:
    # b is renamed away (-> a) as a is renamed in (-> b): no collision.
    return df.rename({"a": "b", "b": "a"})


def rename_fresh(df: DataFrame[In]) -> pl.DataFrame:
    return df.rename({"a": "z"})


def with_columns_overwrite(df: DataFrame[In]) -> pl.DataFrame:
    # Overwriting the existing 'a' is legal — with_columns replaces it.
    return df.with_columns(a=pl.col("b").cast(pl.Int64))


def select_distinct(df: DataFrame[In]) -> pl.DataFrame:
    return df.select(pl.col("a"), pl.col("b").alias("c"))
