"""Error: Nullable type passed where non-nullable expected."""
import polars as pl
from polypolarism import DF


def expects_non_nullable(
    df: DF["{id: Int64, value: Int64}"]
) -> DF["{id: Int64, value: Int64}"]:
    """Expects non-nullable Int64."""
    return df


def caller(
    data: DF["{id: Int64, value: Int64?}"]
) -> DF["{id: Int64, value: Int64}"]:
    """Error: value is Nullable[Int64], mismatches expected Int64."""
    return expects_non_nullable(data)  # Error: nullable mismatch
