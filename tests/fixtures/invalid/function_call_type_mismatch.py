"""Error: Column type mismatch in function argument."""
import polars as pl
from polypolarism import DF


def expects_int_id(df: DF["{id: Int64, value: Float64}"]) -> DF["{id: Int64, value: Float64}"]:
    """id: Int64 を期待する関数."""
    return df


def caller(data: DF["{id: Utf8, value: Float64}"]) -> DF["{id: Int64, value: Float64}"]:
    """Error: id の型が Utf8 で Int64 と不一致."""
    return expects_int_id(data)  # Error: type mismatch for column 'id'
