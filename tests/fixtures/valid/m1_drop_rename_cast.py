"""M1: drop / rename / cast reshape the FrameType."""

import pandera.polars as pa
import polars as pl
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
