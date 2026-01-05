"""Error: Untyped function body cannot be analyzed."""
import polars as pl
from polypolarism import DF


def external_function(df):
    """External library call, cannot be analyzed."""
    return some_external_lib.process(df)


def caller(data: DF["{id: Int64}"]) -> DF["{id: Int64}"]:
    """Error when untyped function body cannot be analyzed."""
    return external_function(data)
