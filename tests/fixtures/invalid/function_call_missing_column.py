"""Error: Missing required column in function argument."""
import polars as pl
from polypolarism import DF


def requires_two_columns(
    df: DF["{id: Int64, name: Utf8}"]
) -> DF["{id: Int64, name: Utf8}"]:
    """Function that requires two columns."""
    return df


def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64, name: Utf8}"]:
    """Error: missing column 'name'."""
    return requires_two_columns(data)  # Error: missing column 'name'
