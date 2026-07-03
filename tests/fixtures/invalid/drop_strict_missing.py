"""Invalid: drop of a missing column WITHOUT strict=False still errors (issue #132 control).

The strict=True default raises ``ColumnNotFoundError`` at runtime, so dropping a
column absent from the schema stays a ``pple-column-not-found`` static FAIL. This
control ensures the #132 fix (strict=False no-op) did not weaken the default.
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class NoQ2(pa.DataFrameModel):
    id: str
    q1: int


def drop_strict_missing(df: DataFrame[NoQ2]) -> DataFrame[NoQ2]:
    return df.drop("q2")
