"""Untyped function passthrough: body analysis infers return type."""
import polars as pl
from polypolarism import DF


def untyped_passthrough(df):
    """型アノテーションなし、DataFrame をそのまま返す."""
    return df


def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64}"]:
    """未注釈関数を経由しても、本体解析で型が推論される."""
    return untyped_passthrough(data)
