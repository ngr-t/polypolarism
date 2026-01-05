"""Invalid test case: Join key types do not match."""

import polars as pl

from polypolarism import DF


def type_mismatch_join(
    users: DF["{user_id: Int64, name: Utf8}"],
    orders: DF["{user_id: Utf8, amount: Float64}"],
) -> DF["{user_id: Int64, name: Utf8, amount: Float64}"]:
    """ERROR: user_id is Int64 in users but Utf8 in orders."""
    return users.join(orders, on="user_id", how="inner")
