"""Variable type annotation in function body."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class IdNameSchema(pa.DataFrameModel):
    id: int
    name: str


def get_data():
    """Fetch DataFrame from external source (type inference not possible)."""
    return external_source.fetch()


def process() -> DataFrame[IdNameSchema]:
    """Provide type information via variable annotation."""
    df: DataFrame[IdNameSchema] = get_data()
    return df
