"""Invalid test case: Join key column does not exist."""

import polars as pl

from polypolarism import DF


def bad_join(
    users: DF["{user_id: Int64, name: Utf8}"],
    orders: DF["{order_id: Int64, amount: Float64}"],
) -> DF["{user_id: Int64, name: Utf8, order_id: Int64, amount: Float64}"]:
    """ERROR: 'user_id' column does not exist in orders DataFrame."""
    return users.join(orders, on="user_id", how="inner")
