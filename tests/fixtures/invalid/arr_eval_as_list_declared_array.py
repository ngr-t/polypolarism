"""Invalid: ``arr.eval(..., as_list=True)`` declared with the Array container.

False-negative twin of ``valid/array_dtype`` (the ``eval_array`` function).
Probed on polars 1.41.2: ``as_list=True`` (added in 1.41) de-arrays the
result into ``List(body dtype)`` — for dtype-changing, aggregating and
length-changing bodies alike — so a declaration keeping the receiver's
Array container is wrong.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    id: int
    q: pl.Array(pl.Int64, 4) = pa.Field()


class Out(pa.DataFrameModel):
    id: int
    doubled: pl.Array(pl.Int64, 4) = pa.Field()  # WRONG: as_list=True yields List(Int64)


def eval_as_list(df: DataFrame[In]) -> DataFrame[Out]:
    return df.select(
        "id",
        doubled=pl.col("q").arr.eval(pl.element() * 2, as_list=True),
    )
