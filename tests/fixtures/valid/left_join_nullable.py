"""Valid test case: Left join with nullable right columns."""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class UsersSchema(pa.DataFrameModel):
    user_id: int
    name: str


class OrdersSchema(pa.DataFrameModel):
    user_id: int
    total: pl.Float64


class UsersWithOptionalOrdersSchema(pa.DataFrameModel):
    user_id: int
    name: str
    total: pl.Float64 = pa.Field(nullable=True)


def users_with_optional_orders(
    users: DataFrame[UsersSchema],
    orders: DataFrame[OrdersSchema],
) -> DataFrame[UsersWithOptionalOrdersSchema]:
    """Left join makes right-side columns nullable."""
    return users.join(orders, on="user_id", how="left")
