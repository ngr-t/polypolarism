"""False-negative twin of ``valid/null_count_schema``.

null_count maps EVERY column to UInt32 (probed, polars 1.37.0-1.41.2);
declaring the tally with the input column's own dtype must fail.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class Events(pa.DataFrameModel):
    id: int
    label: str = pa.Field(nullable=True)


class NullTallyWrong(pa.DataFrameModel):
    id: pl.Int64  # WRONG: null_count returns UInt32 for every column
    label: pl.UInt32

    class Config:
        strict = True


def tally(data: DataFrame[Events]) -> DataFrame[NullTallyWrong]:
    return data.null_count()
