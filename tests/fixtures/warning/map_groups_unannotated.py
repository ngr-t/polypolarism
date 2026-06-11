"""group_by(...).map_groups emits PLW005 (issue #87).

The output schema depends on the group function's body — statically
unknowable, same family as pivot/to_dummies — so instead of the generic
"Could not infer return type" the user is nudged toward assigning the
result to a ``DataFrame[Schema]``-annotated variable (the escape hatch
shown in ``escaped``, which passes via the AnnAssign path).
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Src(pa.DataFrameModel):
    g: str
    v: int


class Out(pa.DataFrameModel):
    g: str
    v: int
    n: pl.Int32


# Deliberately unannotated: a bare ``pl.DataFrame`` annotation would opt the
# helper into open-frame checking (ADR-0006) with a clean (warning-free)
# verdict, which the warning-category invariant rejects.
def head_with_count(gdf):
    return gdf.head(1).with_columns(n=pl.lit(1, dtype=pl.Int32))


def unannotated(df: DataFrame[Src]):
    # No annotation, no inference — the schema exists only in the group
    # function's body.
    return df.group_by("g").map_groups(head_with_count)


def escaped(df: DataFrame[Src]) -> DataFrame[Out]:
    result: DataFrame[Out] = df.group_by("g").map_groups(head_with_count)
    return result
