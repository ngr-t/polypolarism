"""Valid test case: Left join with nullable right columns."""

import polars as pl

from polypolarism import DF


def users_with_optional_orders(
    users: DF["{user_id: Int64, name: Utf8}"],
    orders: DF["{user_id: Int64, total: Float64}"],
) -> DF["{user_id: Int64, name: Utf8, total: Float64?}"]:
    """Left join makes right-side columns nullable."""
    return users.join(orders, on="user_id", how="left")
