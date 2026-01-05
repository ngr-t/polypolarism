"""Variable type annotation in function body."""
import polars as pl
from polypolarism import DF


def get_data():
    """Fetch DataFrame from external source (type inference not possible)."""
    return external_source.fetch()


def process() -> DF["{id: Int64, name: Utf8}"]:
    """Provide type information via variable annotation."""
    df: DF["{id: Int64, name: Utf8}"] = get_data()
    return df
