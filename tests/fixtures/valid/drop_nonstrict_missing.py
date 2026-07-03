"""Valid: drop(strict=False) of a missing column is a legal no-op (issue #132).

``DataFrame.drop(name, strict=False)`` is polars' "drop if present" idiom — a
target absent from the (closed) schema is simply not removed, not a
``pple-column-not-found`` error. The strict=True default still raises (control:
invalid/drop_strict_missing).
"""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class NoQ2(pa.DataFrameModel):
    id: str
    q1: int


def drop_nonstrict_missing(df: DataFrame[NoQ2]) -> DataFrame[NoQ2]:
    # "q2" is not in the schema; strict=False makes this a legal no-op.
    return df.drop("q2", strict=False)
