"""Error: Column type mismatch in function argument."""
import polars as pl
from polypolarism import DF


def expects_int_id(df: DF["{id: Int64, value: Float64}"]) -> DF["{id: Int64, value: Float64}"]:
    """Function that expects id: Int64."""
    return df


def caller(data: DF["{id: Utf8, value: Float64}"]) -> DF["{id: Int64, value: Float64}"]:
    """Error: id type is Utf8, mismatches expected Int64."""
    return expects_int_id(data)  # Error: type mismatch for column 'id'
