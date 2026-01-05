"""Variable type annotation in function body."""
import polars as pl
from polypolarism import DF


def get_data():
    """外部から DataFrame を取得（型推論不可）."""
    return external_source.fetch()


def process() -> DF["{id: Int64, name: Utf8}"]:
    """変数に型アノテーションをつけて型情報を与える."""
    df: DF["{id: Int64, name: Utf8}"] = get_data()
    return df
