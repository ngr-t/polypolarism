"""Variable annotation followed by method chain."""

import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame


class InSchema(pa.DataFrameModel):
    id: int
    value: int


class OutSchema(pa.DataFrameModel):
    id: int
    doubled: int


def process() -> DataFrame[OutSchema]:
    """Continue method chain after variable annotation."""
    df: DataFrame[InSchema] = get_external_data()
    result = df.select(
        pl.col("id"),
        (pl.col("value") * 2).alias("doubled"),
    )
    return result
