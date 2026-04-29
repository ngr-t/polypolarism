"""M1: drop / rename / cast reshape the FrameType."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class In(pa.DataFrameModel):
    id: int
    value: pl.Float64
    name: str


class Out(pa.DataFrameModel):
    user_id: pl.Int32
    value: pl.Float64


def reshape(df: DataFrame[In]) -> DataFrame[Out]:
    return df.drop("name").rename({"id": "user_id"}).cast({"user_id": pl.Int32})
