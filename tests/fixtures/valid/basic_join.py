"""Valid test case: Basic inner join."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class UsersSchema(pa.DataFrameModel):
    user_id: int
    name: str
    country: str


class OrdersSchema(pa.DataFrameModel):
    order_id: int
    user_id: int
    amount: pl.Float64


class JoinedSchema(pa.DataFrameModel):
    user_id: int
    name: str
    country: str
    order_id: int
    amount: pl.Float64


def merge_users_orders(
    users: DataFrame[UsersSchema],
    orders: DataFrame[OrdersSchema],
) -> DataFrame[JoinedSchema]:
    """Join users and orders on user_id."""
    return users.join(orders, on="user_id", how="inner")
