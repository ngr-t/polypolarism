"""Error: Nullable type passed where non-nullable expected."""
import polars as pl
from polypolarism import DF


def expects_non_nullable(
    df: DF["{id: Int64, value: Int64}"]
) -> DF["{id: Int64, value: Int64}"]:
    """Non-nullable Int64 を期待."""
    return df


def caller(
    data: DF["{id: Int64, value: Int64?}"]
) -> DF["{id: Int64, value: Int64}"]:
    """Error: value が Nullable[Int64] で Int64 と不一致."""
    return expects_non_nullable(data)  # Error: nullable mismatch
