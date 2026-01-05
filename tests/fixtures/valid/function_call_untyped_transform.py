"""Untyped function with transformation: body analysis infers return type."""
import polars as pl
from polypolarism import DF


def untyped_add_column(df):
    """型アノテーションなし、列を追加して返す."""
    return df.with_columns(pl.lit(100).alias("new_col"))


def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64, new_col: Int64}"]:
    """未注釈関数を経由しても、本体解析で変換後の型が推論される."""
    return untyped_add_column(data)
