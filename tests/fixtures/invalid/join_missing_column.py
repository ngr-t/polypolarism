"""Invalid test case: Join key column does not exist."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class UsersSchema(pa.DataFrameModel):
    user_id: int
    name: str


class OrdersSchema(pa.DataFrameModel):
    order_id: int
    amount: pl.Float64


class JoinedSchema(pa.DataFrameModel):
    user_id: int
    name: str
    order_id: int
    amount: pl.Float64


def bad_join(
    users: DataFrame[UsersSchema],
    orders: DataFrame[OrdersSchema],
) -> DataFrame[JoinedSchema]:
    """ERROR: 'user_id' column does not exist in orders DataFrame."""
    return users.join(orders, on="user_id", how="inner")
