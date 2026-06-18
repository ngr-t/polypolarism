"""unique subset column doesn't exist (issue #35, pple-column-not-found)."""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class ASB(pa.DataFrameModel):
    a: int
    s: str
    b: int

    class Config:
        coerce = True


@pa.check_types
def bug_unique_subset_ghost(df: DataFrame[ASB]) -> DataFrame[ASB]:
    return df.unique(subset=["ghost"])  # 'ghost' doesn't exist
