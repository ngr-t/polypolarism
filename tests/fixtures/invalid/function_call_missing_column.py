"""Error: Missing required column in function argument."""
import polars as pl
from polypolarism import DF


def requires_two_columns(
    df: DF["{id: Int64, name: Utf8}"]
) -> DF["{id: Int64, name: Utf8}"]:
    """2カラムを必要とする関数."""
    return df


def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64, name: Utf8}"]:
    """Error: name カラムが不足."""
    return requires_two_columns(data)  # Error: missing column 'name'
