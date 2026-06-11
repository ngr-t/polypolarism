"""Invalid: declaring a right-side column non-nullable after a left join.

False-negative twin of ``valid/left_join_nullable.py``: the left join makes
``total`` Nullable[Float64], so a declared non-nullable ``total`` must be
rejected — this guards the join-nullability rule against silently regressing
to accept-anything.
"""

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame


class UsersSchema(pa.DataFrameModel):
    user_id: int
    name: str


class OrdersSchema(pa.DataFrameModel):
    user_id: int
    total: pl.Float64


class UsersWithOrdersSchema(pa.DataFrameModel):
    user_id: int
    name: str
    total: pl.Float64  # WRONG: left join makes the right side nullable


def users_with_orders(
    users: DataFrame[UsersSchema],
    orders: DataFrame[OrdersSchema],
) -> DataFrame[UsersWithOrdersSchema]:
    return users.join(orders, on="user_id", how="left")
