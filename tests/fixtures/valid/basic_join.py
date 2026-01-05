"""Valid test case: Basic inner join."""

import polars as pl

from polypolarism import DF


def merge_users_orders(
    users: DF["{user_id: Int64, name: Utf8, country: Utf8}"],
    orders: DF["{order_id: Int64, user_id: Int64, amount: Float64}"],
) -> DF["{user_id: Int64, name: Utf8, country: Utf8, order_id: Int64, amount: Float64}"]:
    """Join users and orders on user_id."""
    return users.join(orders, on="user_id", how="inner")
