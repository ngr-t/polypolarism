"""Variable annotation followed by method chain."""
import polars as pl
from polypolarism import DF


def process() -> DF["{id: Int64, doubled: Int64}"]:
    """変数に型アノテーション後、メソッドチェーンを続ける."""
    df: DF["{id: Int64, value: Int64}"] = get_external_data()
    result = df.select(
        pl.col("id"),
        (pl.col("value") * 2).alias("doubled"),
    )
    return result
