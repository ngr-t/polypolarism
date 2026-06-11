"""`PLY033`: a variable annotation that re-interprets an inferable RHS as
an unrelated type is an error (ADR-0005 two-direction rule).

False-negative twin of ``valid/variable_annotation_basic``: there the RHS
is uninferable (external fetch) and the annotation legitimately provides
the type; here the RHS is precisely inferable and the annotation provably
contradicts it — the discovered N-1 false negative.
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    id: int
    name: str


class WrongView(pa.DataFrameModel):
    id: int
    name: float  # WRONG: 'name' provably infers Utf8; Float64 is unrelated


def relabel(df: DataFrame[In]) -> DataFrame[In]:
    view: DataFrame[WrongView] = df.select("id", "name")  # noqa: F841
    return df
