"""M1 invalid: rename source column doesn't exist."""

import pandera.polars as pa
from pandera.typing.polars import DataFrame


class S(pa.DataFrameModel):
    id: int


def f(df: DataFrame[S]):
    return df.rename({"nope": "x"})
