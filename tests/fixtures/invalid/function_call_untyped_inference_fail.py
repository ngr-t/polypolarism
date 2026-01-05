"""Error: Untyped function body cannot be analyzed."""
import polars as pl
from polypolarism import DF


def external_function(df):
    """外部ライブラリ呼び出しなど、解析不能な処理."""
    return some_external_lib.process(df)


def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64}"]:
    """未注釈関数の本体が解析できない場合はエラー."""
    return external_function(data)
