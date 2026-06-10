"""sort key column doesn't exist (issue #29, PLY007)."""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class AInt(pa.DataFrameModel):
    a: int

    class Config:
        coerce = True


@pa.check_types
def bug_sort_nonexistent(df: DataFrame[AInt]) -> DataFrame[AInt]:
    return df.sort("ghost")  # 'ghost' doesn't exist
