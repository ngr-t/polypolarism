"""Invalid test case: Join key types do not match."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class UsersSchema(pa.DataFrameModel):
    user_id: int
    name: str


class OrdersSchema(pa.DataFrameModel):
    user_id: str
    amount: pl.Float64


class JoinedSchema(pa.DataFrameModel):
    user_id: int
    name: str
    amount: pl.Float64


def type_mismatch_join(
    users: DataFrame[UsersSchema],
    orders: DataFrame[OrdersSchema],
) -> DataFrame[JoinedSchema]:
    """ERROR: user_id is Int64 in users but Utf8 in orders."""
    return users.join(orders, on="user_id", how="inner")
