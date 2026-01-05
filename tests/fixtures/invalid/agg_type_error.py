"""Invalid test case: Aggregation function type error."""

import polars as pl

from polypolarism import DF


def sum_on_string(
    data: DF["{category: Utf8, label: Utf8}"],
) -> DF["{category: Utf8, label_sum: Utf8}"]:
    """ERROR: Cannot apply sum() to Utf8 column."""
    return data.group_by("category").agg(
        pl.col("label").sum().alias("label_sum"),
    )
